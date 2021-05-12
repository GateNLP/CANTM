from transformers import BertModel
import math
import os
import torch.nn.functional as F
import torch
import torch.nn as nn


class SingleHeadAttention(nn.Module):
    def __init__(self, d_model, d_output, dropout = 0.1):
        super().__init__()

        self.q = nn.Parameter(torch.randn([d_output, 1]).float())
        self.v_linear = nn.Linear(d_model, d_output)
        self.dropout_v = nn.Dropout(dropout)
        self.k_linear = nn.Linear(d_model, d_output)
        self.dropout_k = nn.Dropout(dropout)
        self.softmax_simi = nn.Softmax(dim=1)

        self.dropout = nn.Dropout(dropout)
        #self.out = nn.Linear(d_output, d_output)

    def forward(self, x, mask=None):
        k = self.k_linear(x)
        k = F.relu(k)
        k = self.dropout_k(k)
        v = self.v_linear(x)
        v = F.relu(v)
        v = self.dropout_v(v)

        dotProducSimi = k.matmul(self.q)
        normedSimi = self.softmax_simi(dotProducSimi)
        attVector = v.mul(normedSimi)
        weightedSum = torch.sum(attVector, dim=1)
        #output = self.out(weightedSum)
        return weightedSum


class Norm(nn.Module):
    def __init__(self, d_model, eps = 1e-6):
        super().__init__()

        self.size = d_model

        # create two learnable parameters to calibrate normalisation
        self.alpha = nn.Parameter(torch.ones(self.size))
        self.bias = nn.Parameter(torch.zeros(self.size))

        self.eps = eps

    def forward(self, x):
        norm = self.alpha * (x - x.mean(dim=-1, keepdim=True)) \
        / (x.std(dim=-1, keepdim=True) + self.eps) + self.bias
        return norm



def attention(q, k, v, d_k, mask=None, dropout=None):

    scores = torch.matmul(q, k.transpose(-2, -1)) /  math.sqrt(d_k)

    if mask is not None:
        mask = mask.unsqueeze(1)
        scores = scores.masked_fill(mask == 0, -1e9)

    scores = F.softmax(scores, dim=-1)

    if dropout is not None:
        scores = dropout(scores)

    output = torch.matmul(scores, v)
    return output



class EncoderLayer(nn.Module):
    def __init__(self, d_model, heads, dropout=0.1):
        super().__init__()
        self.norm_1 = Norm(d_model)
        self.norm_2 = Norm(d_model)
        self.attn = MultiHeadAttention(heads, d_model, dropout=dropout)
        self.ff = FeedForward(d_model, dropout=dropout)
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)

    def forward(self, x, mask):
        x2 = self.norm_1(x)
        x = x + self.dropout_1(self.attn(x2,x2,x2,mask))
        x2 = self.norm_2(x)
        x = x + self.dropout_2(self.ff(x2))
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, heads, d_model, dropout = 0.1):
        super().__init__()
        
        self.d_model = d_model
        self.d_k = d_model // heads
        self.h = heads
        
        self.q_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(d_model, d_model)
    
    def forward(self, q, k, v, mask=None):
        
        bs = q.size(0)
        
        # perform linear operation and split into N heads
        # bs, sl, d_model --> bs, sl, heads, sub_d_model
        # d_model = heads * sub_d_model
        k = self.k_linear(k).view(bs, -1, self.h, self.d_k)
        q = self.q_linear(q).view(bs, -1, self.h, self.d_k)
        v = self.v_linear(v).view(bs, -1, self.h, self.d_k)
        
        # transpose to get dimensions bs * N * sl * d_model
        k = k.transpose(1,2)
        q = q.transpose(1,2)
        v = v.transpose(1,2)
        

        # calculate attention using function we will define next
        scores = attention(q, k, v, self.d_k, mask, self.dropout)
        # concatenate heads and put through final linear layer
        concat = scores.transpose(1,2).contiguous()\
        .view(bs, -1, self.d_model)
        output = self.out(concat)
    
        return output


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff=1024, dropout = 0.1):
        super().__init__()

        # We set d_ff as a default to 2048
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        x = self.dropout(F.relu(self.linear_1(x)))
        x = self.linear_2(x)
        return x


class BERT_Embedding(nn.Module):
    def __init__(self, config):
        super().__init__()

        bert_model_path = os.path.join(config['BERT'].get('bert_path'), 'model')
        self.bert_dim = int(config['BERT'].get('bert_dim'))
        self.trainable_layers = config['BERT'].get('trainable_layers')
        self.bert = BertModel.from_pretrained(bert_model_path, output_attentions=True,output_hidden_states=True)
        if self.trainable_layers:
            #print(self.trainable_layers)
            #self.bert = BertModel.from_pretrained(bert_model_path)
            for name, param in self.bert.named_parameters():
                if name in self.trainable_layers:
                    param.requires_grad = True
                    #print(name, param)
                else:
                    param.requires_grad = False
        else:
            for p in self.bert.parameters():
                p.requires_grad = False

    def forward(self, x, mask=None):
        if mask == None:
            mask = x != 0
            mask.type(x.type())
        bert_rep = self.bert(x, attention_mask=mask)
        #print(len(bert_rep))
        #print(len(bert_rep[3]))
        #print(bert_rep[3][11].shape)
        return bert_rep


class Dense(nn.Module):
    def __init__(self, input_dim, out_dim, non_linear=None):
        super().__init__()
        self.dense = nn.Linear(input_dim, out_dim)
        self.non_linear = non_linear

    def forward(self, x):
        output = self.dense(x)
        if self.non_linear:
            output = self.non_linear(output)
        return output



class Topics(nn.Module):
    def __init__(self, k, vocab_size, bias=True):
        super(Topics, self).__init__()
        self.k = k
        self.vocab_size = vocab_size
        self.topic = nn.Linear(k, vocab_size, bias=bias)

    def forward(self, logit):
        # return the log_prob of vocab distribution
        return torch.log_softmax(self.topic(logit), dim=-1)

    def get_topics(self):
        #print('hey')
        #print(self.topic.weight)
        return torch.softmax(self.topic.weight.data.transpose(0, 1), dim=-1)

    def get_topic_word_logit(self):
        """topic x V.
        Return the logits instead of probability distribution
        """
        return self.topic.weight.transpose(0, 1)


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, *input):
        if len(input) == 1:
            return input[0]
        return input


def kld(mu, log_sigma):
    """log q(z) || log p(z).
    mu: batch_size x dim
    log_sigma: batch_size x dim
    """
    return -0.5 * (1 - mu ** 2 + 2 * log_sigma - torch.exp(2 * log_sigma)).sum(dim=-1)

class BERT_Mapping_mapping(nn.Module):
    def __init__(self, bert_dim):
        super().__init__()
        self.att = SingleHeadAttention(bert_dim, bert_dim)

    def forward(self,x):
        atted = self.att(x)
        return atted


class WVHidden(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.hidden1 = nn.Linear(input_dim, hidden_dim)

    def forward(self, x):
        hidden = F.leaky_relu(self.hidden1(x))
        return hidden


class WVClassifier(nn.Module):
    def __init__(self, n_hidden, n_classes):
        super().__init__()
        self.layer_output = torch.nn.Linear(n_hidden, n_classes)

    def forward(self, x):
        out = self.layer_output(x)
        return out


class CLSAW_TopicModel_Base(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        self._init_params()
        if config:
            self._read_config(config)

    def _init_params(self):
        self.hidden_dim = 300
        self.z_dim = 100
        self.ntopics = 50
        self.class_topic_loss_lambda = 1
        self.classification_loss_lambda = 1
        self.banlance_loss = False

    def _read_config(self, config):
        self.n_classes = len(config['TARGET'].get('labels'))
        if 'MODEL' in config:
            if 'hidden_dim' in config['MODEL']:
                self.hidden_dim = int(config['MODEL'].get('hidden_dim'))
            if 'z_dim' in config['MODEL']:
                self.z_dim = int(config['MODEL'].get('z_dim'))
            if 'ntopics' in config['MODEL']:
                self.ntopics = int(config['MODEL'].get('ntopics'))
            if 'class_topic_loss_lambda' in config['MODEL']:
                self.class_topic_loss_lambda = float(config['MODEL'].get('class_topic_loss_lambda'))
            if 'classification_loss_lambda' in config['MODEL']:
                self.class_topic_loss_lambda = float(config['MODEL'].get('classification_loss_lambda'))
            if 'banlance_loss' in config['MODEL']:
                self.banlance_loss = config['MODEL'].as_bool('banlance_loss')
        self.n_class_topics = self.z_dim+self.n_classes

        





