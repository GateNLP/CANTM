import torch
from torch import nn
from torch.nn import init
from torch.nn import functional as F
import math
from .miscLayer import BERT_Embedding, WVHidden, WVClassifier, Identity, Topics, kld, CLSAW_TopicModel_Base


class NVDM(CLSAW_TopicModel_Base):
    def __init__(self, config, vocab_dim=None):
        super().__init__(config=config)
        default_config = {}

        self.bert_embedding = BERT_Embedding(config)
        bert_dim = self.bert_embedding.bert_dim

        if self.banlance_loss:
            self.banlance_lambda = float(math.ceil(vocab_dim/self.n_classes))
        else:
            self.banlance_lambda = 1

        #self.wv_hidden = WVHidden(bert_dim, self.hidden_dim)
        self.hidden_dim = bert_dim

        ##############M1###########################################
        self.mu_z1 = nn.Linear(self.hidden_dim, self.z_dim)
        self.log_sigma_z1 = nn.Linear(self.hidden_dim, self.z_dim)
        self.x_only_topics = Topics(self.z_dim, vocab_dim)
        #self.xy_classifier = WVClassifier(self.z_dim, self.n_classes)
        #self.class_criterion = nn.CrossEntropyLoss()


        #############M2############################################
        #self.hidden_y_dim = self.hidden_dim + self.n_classes
        #self.z_y_dim = self.z_dim + self.n_classes
        #self.x_y_hidden = WVHidden(self.hidden_y_dim, self.hidden_dim)
        #self.z_y_hidden = WVHidden(self.z_y_dim, self.ntopics)

        #self.mu_z2 = nn.Linear(self.hidden_dim, self.z_dim)
        #self.log_sigma_z2 = nn.Linear(self.hidden_dim, self.z_dim)
        #self.xy_topics = Topics(self.ntopics, vocab_dim)
        #self.z2y_classifier = WVClassifier(self.ntopics, self.n_classes)

        ############################################################
        self.h_to_z = Identity()
        #self.class_topics = Topics(self.n_classes, vocab_dim)
        self.reset_parameters()


    def forward(self,x, mask=None, n_samples=1, bow=None, train=False, true_y=None, pre_embd=False, true_y_ids=None):
        #print(true_y.shape)
        if pre_embd:
            bert_rep = x
        else:
            bert_rep = self.bert_embedding(x, mask)
            bert_rep = bert_rep[0]
        atted = bert_rep[:,0]
        #hidden = self.wv_hidden(atted)
        hidden = atted
        mu_z1 = self.mu_z1(hidden)
        log_sigma_z1 = self.log_sigma_z1(hidden)

        kldz1 = kld(mu_z1, log_sigma_z1)
        rec_loss_z1 = 0
        classifier_loss = 0
        kldz2 = 0
        rec_loss_z2 = 0
        log_y_hat_rec_loss = 0
        class_topic_rec_loss = 0


        for i in range(n_samples):
            z1 = torch.zeros_like(mu_z1).normal_() * torch.exp(log_sigma_z1) + mu_z1
            z1 = self.h_to_z(z1)
            log_probz_1 = self.x_only_topics(z1)
            rec_loss_z1 = rec_loss_z1-(log_probz_1 * bow).sum(dim=-1)


        rec_loss_z1 = rec_loss_z1/n_samples
        elbo_z1 = kldz1 + rec_loss_z1
        total_loss = elbo_z1.sum() 

        y_hat_logis = torch.zeros(x.shape[0], self.n_classes)
        elbo_z2 = torch.zeros_like(elbo_z1)
        classifier_loss = torch.tensor(0)


        y = {
            'loss': total_loss,
            'elbo_xy': elbo_z2,
            'rec_loss': rec_loss_z2,
            'kld': kldz2,
            'cls_loss': classifier_loss,
            'class_topic_loss': class_topic_rec_loss,
            'y_hat': y_hat_logis,
            'elbo_x': elbo_z1
        }

        return y, None


    def reset_parameters(self):
        init.zeros_(self.log_sigma_z1.weight)
        init.zeros_(self.log_sigma_z1.bias)

    def get_topics(self):
        return self.x_only_topics.get_topics()

    def get_class_topics(self):
        return self.x_only_topics.get_topics()

    def get_x_only_topics(self):
        return self.x_only_topics.get_topics()


