from transformers import BertModel
from .miscLayer import BERT_Embedding, CLSAW_TopicModel_Base, WVHidden
import os
import torch.nn.functional as F
import torch
import torch.nn as nn


class BERT_Simple(CLSAW_TopicModel_Base):
    def __init__(self, config, **kwargs):
        super().__init__(config=config)

        self.bert_embedding = BERT_Embedding(config)
        bert_dim = self.bert_embedding.bert_dim

        self.hidden_dim = bert_dim
        self.hidden2 = WVHidden(self.hidden_dim, self.z_dim)
        self.layer_output = torch.nn.Linear(self.z_dim, self.n_classes)
        #self.layer_output = torch.nn.Linear(bert_dim, self.n_classes)


    def forward(self, x, mask=None, pre_embd=False):
        if pre_embd:
            bert_rep = x
        else:
            bert_rep = self.bert_embedding(x, mask)
            bert_rep = bert_rep[0]
        bert_rep = bert_rep[:,0]

        #hidden = self.hidden1(bert_rep)
        hidden = bert_rep
        hidden = self.hidden2(hidden)
        out = self.layer_output(hidden)

        y = {
            'y_hat':out
        }

        return y

