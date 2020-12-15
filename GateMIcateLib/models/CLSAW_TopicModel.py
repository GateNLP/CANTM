import torch
from torch import nn
from torch.nn import init
from torch.nn import functional as F
import math
from .miscLayer import BERT_Embedding, WVHidden, WVClassifier, Identity, Topics, kld, CLSAW_TopicModel_Base


class CLSAW_TopicModel(CLSAW_TopicModel_Base):
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
        self.xy_classifier = WVClassifier(self.z_dim, self.n_classes)
        self.class_criterion = nn.CrossEntropyLoss()


        #############M2############################################
        self.hidden_y_dim = self.hidden_dim + self.n_classes
        self.z_y_dim = self.z_dim + self.n_classes
        self.x_y_hidden = WVHidden(self.hidden_y_dim, self.hidden_dim)
        self.z_y_hidden = WVHidden(self.z_y_dim, self.ntopics)

        self.mu_z2 = nn.Linear(self.hidden_dim, self.z_dim)
        self.log_sigma_z2 = nn.Linear(self.hidden_dim, self.z_dim)
        self.xy_topics = Topics(self.ntopics, vocab_dim)
        self.z2y_classifier = WVClassifier(self.ntopics, self.n_classes)

        ############################################################
        self.h_to_z = Identity()
        self.class_topics = Topics(self.n_classes, vocab_dim)
        self.reset_parameters()


    def forward(self,x, mask=None, n_samples=1, bow=None, train=False, true_y=None, pre_embd=False, true_y_ids=None, update_catopic=False):
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


        #if not train:
        #    ### for discriminator, we only use mean
        #    z1 = mu_z1
        #    y_hat_logis = self.xy_classifier(z1)
        #    log_probz_1 = self.x_only_topics(z1)
        #    y_hat = torch.softmax(y_hat_logis, dim=-1)
        #    log_prob_class_topic = self.class_topics(y_hat)
        #    #y = y_hat_logis


        for i in range(n_samples):
            z1 = torch.zeros_like(mu_z1).normal_() * torch.exp(log_sigma_z1) + mu_z1
            z1 = self.h_to_z(z1)
            log_probz_1 = self.x_only_topics(z1)


            #if train or update_catopic:
            #    z1 = torch.zeros_like(mu_z1).normal_() * torch.exp(log_sigma_z1) + mu_z1
            #    z1 = self.h_to_z(z1)
            #    log_probz_1 = self.x_only_topics(z1)
            #    y_hat_logis = self.xy_classifier(z1)
            #    y_hat = torch.softmax(y_hat_logis, dim=-1)
            #    log_prob_class_topic = self.class_topics(y_hat)

            if train or update_catopic:
                y_hat_logis = self.xy_classifier(z1)
                y_hat = torch.softmax(y_hat_logis, dim=-1)
                #print(y_hat.shape)
            else:
                y_hat_logis = self.xy_classifier(mu_z1)
                y_hat = torch.softmax(y_hat_logis, dim=-1)

            if train:
                classifier_loss += self.class_criterion(y_hat_logis, true_y_ids)

            log_prob_class_topic = self.class_topics(y_hat)


            y_hat_h = torch.cat((hidden, y_hat), dim=-1)
            x_y_hidden = self.x_y_hidden(y_hat_h)
            mu_z2 = self.mu_z2(x_y_hidden)
            log_sigma_z2 = self.log_sigma_z2(x_y_hidden)
            z2 = torch.zeros_like(mu_z2).normal_() * torch.exp(log_sigma_z2) + mu_z2
            y_hat_z = torch.cat((z2, y_hat), dim=-1)
            topic = self.z_y_hidden(y_hat_z)
            log_prob_z2 = self.xy_topics(topic)
            y_hat_rec = self.z2y_classifier(topic)
            log_y_hat_rec = torch.log_softmax(y_hat_rec, dim=-1)

            rec_loss_z1 = rec_loss_z1-(log_probz_1 * bow).sum(dim=-1)
            kldz2 += kld(mu_z2, log_sigma_z2)
            rec_loss_z2 = rec_loss_z2 - (log_prob_z2 * bow).sum(dim=-1)

            #log_y_hat_rec_loss = log_y_hat_rec_loss - (log_y_hat_rec*true_y).sum(dim=-1)
            log_y_hat_rec_loss = log_y_hat_rec_loss - (log_y_hat_rec*y_hat).sum(dim=-1)
            class_topic_rec_loss = class_topic_rec_loss - (log_prob_class_topic*bow).sum(dim=-1)


        rec_loss_z1 = rec_loss_z1/n_samples
        #print(rec_loss_z1.shape)
        classifier_loss = classifier_loss/n_samples
        kldz2 = kldz2/n_samples
        rec_loss_z2 = rec_loss_z2/n_samples
        log_y_hat_rec_loss = log_y_hat_rec_loss/n_samples
        class_topic_rec_loss = class_topic_rec_loss/n_samples

        elbo_z1 = kldz1 + rec_loss_z1
        #print(elbo_z1.shape)
        #elbo_z1 = elbo_z1.sum()
        elbo_z2 = kldz2 + rec_loss_z2 + log_y_hat_rec_loss
        #print(elbo_z2)
        #elbo_z2 = elbo_z2.sum()
        #class_topic_rec_loss = class_topic_rec_loss.sum()
        classifier_loss = classifier_loss

        total_loss = elbo_z1.sum() + elbo_z2.sum() + class_topic_rec_loss.sum() + classifier_loss*self.banlance_lambda*self.classification_loss_lambda

        if update_catopic:
            total_loss = elbo_z2.sum()



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
####################################################################################################################################################
#        else:
#            z1 = mu_z1
#            y_hat_logis = self.xy_classifier(z1)
#            y_hat = torch.softmax(y_hat_logis, dim=-1)
#            y = y_hat_logis
#
#
#            y_hat_h = torch.cat((hidden, y_hat), dim=-1)
#            x_y_hidden = self.x_y_hidden(y_hat_h)
#            mu_z2 = self.mu_z2(x_y_hidden)
#            log_sigma_z2 = self.log_sigma_z2(x_y_hidden)
#            z2 = torch.zeros_like(mu_z2).normal_() * torch.exp(log_sigma_z2) + mu_z2
#
#            kldz2 = kld(mu_z2, log_sigma_z2)
#            log_prob_z2 = self.xy_topics(z2)
#            y_hat_rec = self.z2y_classifier(z2)
#            log_y_hat_rec = torch.log_softmax(y_hat_rec, dim=-1)
#
#


        return y, None


    def reset_parameters(self):
        init.zeros_(self.log_sigma_z1.weight)
        init.zeros_(self.log_sigma_z1.bias)
        init.zeros_(self.log_sigma_z2.weight)
        init.zeros_(self.log_sigma_z2.bias)



    def get_topics(self):
        return self.xy_topics.get_topics()

    def get_class_topics(self):
        return self.class_topics.get_topics()

    def get_x_only_topics(self):
        return self.x_only_topics.get_topics()


