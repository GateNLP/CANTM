import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import copy
import os
from pathlib import Path
import pickle
import datetime
from .modelUlti import modelUlti

class ModelUltiClass(modelUlti):
    def __init__(self, net=None, gpu=False, load_path=None):
        super().__init__(net=net, gpu=gpu)
        if load_path:
            self.loadModel(load_path)
            if self.gpu:
                self.net.cuda()


    def train(self, trainBatchIter, num_epohs=100, valBatchIter=None, cache_path=None, earlyStopping='cls_loss', patience=5):
        pytorch_total_params = sum(p.numel() for p in self.net.parameters())
        print('total_params: ',pytorch_total_params)
        pytorch_train_params = sum(p.numel() for p in self.net.parameters() if p.requires_grad)
        print('train_params: ',pytorch_train_params)

        self.bowdict = trainBatchIter.dataIter.postProcessor.dictProcess
        self.labels = trainBatchIter.dataIter.postProcessor.labelsFields

        if earlyStopping == 'None':
            earlyStopping = None
        self.cache_path = cache_path
        output_dict = {}
        output_dict['accuracy'] = 'no val iter'
        output_dict['perplexity'] = 'no val iter'
        output_dict['perplexity_x_only'] = 'no val iter'

        self.evaluation_history = []
        self.optimizer = optim.Adam(self.net.parameters())
        print(num_epohs)
        for epoch in range(num_epohs):
            begin_time = datetime.datetime.now()
            all_loss = []
            all_elboz1 = []
            all_elboz2 = []
            all_bow = []
            trainIter = self.pred(trainBatchIter, train=True)
            for current_prediction in trainIter:
                self.optimizer.zero_grad()

                pred = current_prediction['pred']
                y = current_prediction['y']
                atted = current_prediction['atted'] 
                loss = pred['loss']
                cls_loss = pred['cls_loss'].sum()
                elbo_z1 = pred['elbo_x'].to('cpu').detach().numpy()
                elbo_z2 = pred['elbo_xy'].to('cpu').detach().numpy()
                bow_x = current_prediction['x_bow'].to('cpu').detach().numpy()

                all_elboz1.append(elbo_z1)
                all_elboz2.append(elbo_z2)
                all_bow.append(bow_x)

                loss.backward()
                self.optimizer.step()
                loss_value = float(cls_loss.data.item())
                all_loss.append(loss_value)
                all_elboz1
            if epoch % 3 == 0:
                topics = self.getTopics(trainBatchIter.dataIter.postProcessor.dictProcess, cache_path=self.cache_path)
                #print('===========')
                x_only_topic = self.get_x_only_Topics(trainBatchIter.dataIter.postProcessor.dictProcess, cache_path=self.cache_path)
                #print('========================')
                self.getClassTopics(trainBatchIter.dataIter.postProcessor.dictProcess, cache_path=self.cache_path)
                cache_last_path = os.path.join(self.cache_path, 'last_net.model')
                self.saveWeights(cache_last_path)
            if valBatchIter:
                output_dict = self.eval(valBatchIter, get_perp=True)

            avg_loss = sum(all_loss)/len(all_loss)
            output_dict['cls_loss'] = -avg_loss
            perplexity_z1, log_perp_z1 = self._get_prep(all_elboz1, all_bow)
            perplexity_z2, log_perp_z2 = self._get_prep(all_elboz2, all_bow)
            output_dict['train_ppl_loss'] = -perplexity_z1
            if earlyStopping:
                stop_signal = self.earlyStop(output_dict, patience=patience, metric=earlyStopping, num_epoch=num_epohs)
                if stop_signal:
                    print('stop signal received, stop training')
                    cache_load_path = os.path.join(self.cache_path, 'best_net.model')
                    print('finish training, load model from ', cache_load_path)
                    self.loadWeights(cache_load_path)
                    break
             
            end_time = datetime.datetime.now()
            timeused = end_time - begin_time
            print('epoch ', epoch, 'loss', avg_loss, ' val acc: ', output_dict['accuracy'], 'test_pplz2: ', output_dict['perplexity'], 'test_perpz1: ', output_dict['perplexity_x_only'], 'train_pplz2: ', perplexity_z2, 'train_perpz1: ', perplexity_z1, 'time: ', timeused)
        cache_last_path = os.path.join(self.cache_path, 'last_net.model')
        self.saveWeights(cache_last_path)
        self.saveModel(self.cache_path)
        self.getTopics(trainBatchIter.dataIter.postProcessor.dictProcess, cache_path=self.cache_path)
        #print('===========')
        self.getClassTopics(trainBatchIter.dataIter.postProcessor.dictProcess, cache_path=self.cache_path)
        #print('===========')
        x_only_topic = self.get_x_only_Topics(trainBatchIter.dataIter.postProcessor.dictProcess, cache_path=self.cache_path)
            
        
    def getClassTopics(self, dictProcess, ntop=10, cache_path=None):
        termMatrix = self.net.get_class_topics()
        topicWordList = []
        for each_topic in termMatrix:
            trans_list = list(enumerate(each_topic.cpu().numpy()))
            #print(trans_list)
            trans_list = sorted(trans_list, key=lambda k: k[1], reverse=True)
            #print(trans_list)
            topic_words = [dictProcess.get(item[0]) for item in trans_list[:ntop]]
            #print(topic_words)
            topicWordList.append(topic_words)
        if cache_path:
            save_path = os.path.join(cache_path, 'classtopics.txt')
            self.saveTopic(topicWordList, save_path)
        return topicWordList



    def saveModel(self, cache_path):
        model_path = os.path.join(cache_path, 'net.model')
        dict_path = os.path.join(cache_path, 'dict.pkl')
        label_path = os.path.join(cache_path, 'label.pkl')

        torch.save(self.net, model_path)
        with open(dict_path, 'wb') as fp:
            pickle.dump(self.bowdict, fp)

        with open(label_path, 'wb') as fp:
            pickle.dump(self.labels, fp)

    def loadModel(self, cache_path):
        model_path = os.path.join(cache_path, 'net.model')
        dict_path = os.path.join(cache_path, 'dict.pkl')
        label_path = os.path.join(cache_path, 'label.pkl')

        self.net = torch.load(model_path, map_location=torch.device("cpu"))
        self.net.eval()
        with open(dict_path, 'rb') as fp:
            self.bowdict = pickle.load(fp)

        with open(label_path, 'rb') as fp:
            self.labels = pickle.load(fp)

    def pred(self, batchGen, train=False, updateTopic=False):
        if train or updateTopic:
            self.net.train()
            #self.optimizer.zero_grad()
        else:
            self.net.eval()
        i=0
        pre_embd = False
        for x, x_bow, y in batchGen:
            i+=1
            print("processing batch", i, end='\r')
            if self.gpu:
                y = y.type(torch.cuda.LongTensor)
                x_bow = x_bow.type(torch.cuda.FloatTensor)
                x_bow.cuda()
                y.cuda()
                if batchGen.dataIter.postProcessor.embd_ready:
                    pre_embd = True
                    x = x.type(torch.cuda.FloatTensor).squeeze(1)
                    x.cuda()
                else:
                    x = x.type(torch.cuda.LongTensor)
                    x.cuda()

            if train:
                one_hot_y = self.y2onehot(y)
                if batchGen.dataIter.label_weights_list:
                    n_samples = self.get_num_samples(y, batchGen.dataIter.label_weights_list)
                else:
                    n_samples = 10
                #print(n_samples)
                pred, atted = self.net(x, bow=x_bow, train=True, true_y=one_hot_y, n_samples=n_samples, pre_embd=pre_embd, true_y_ids=y)
            elif updateTopic:
                pred, atted = self.net(x, bow=x_bow, pre_embd=pre_embd, update_catopic=True)
            else:
                pred, atted = self.net(x, bow=x_bow, pre_embd=pre_embd)
                #pred = pred['y_hat']
            output_dict = {}
            output_dict['pred'] = pred
            output_dict['y'] = y
            output_dict['atted'] = atted
            output_dict['x_bow'] = x_bow

            yield output_dict

    def application_oneSent(self, x):
        if self.gpu:
            x = x.type(torch.cuda.LongTensor)
            x.cuda()


        pred, atted = self.net(x)
        output_dict = {}
        output_dict['pred'] = pred
        output_dict['atted'] = atted
        return output_dict 

    def get_num_samples(self, y, weight_list):
        n_samples = 0
        for y_item in y:
            n_samples += weight_list[y_item.item()]
        return n_samples

    def y2onehot(self, y):
        num_class = self.net.n_classes
        one_hot_y_list = []
        for i in range(len(y)):
            current_one_hot = [0]*num_class
            current_one_hot[y[i].item()] = 1
            one_hot_y_list.append(copy.deepcopy(current_one_hot))
        tensor_one_hot_y = torch.tensor(one_hot_y_list)
        if self.gpu:
            tensor_one_hot_y = tensor_one_hot_y.type(torch.cuda.FloatTensor)
            tensor_one_hot_y = tensor_one_hot_y.cuda()
        return tensor_one_hot_y 

    def getTopics(self, dictProcess, ntop=10, cache_path=None):
        termMatrix = self.net.get_topics()
        #print(termMatrix.shape)
        topicWordList = []
        for each_topic in termMatrix:
            trans_list = list(enumerate(each_topic.cpu().numpy()))
            #print(trans_list)
            trans_list = sorted(trans_list, key=lambda k: k[1], reverse=True)
            #print(trans_list)
            topic_words = [dictProcess.get(item[0]) for item in trans_list[:ntop]]
            #print(topic_words)
            topicWordList.append(topic_words)
        if cache_path:
            save_path = os.path.join(cache_path, 'topics.txt')
            self.saveTopic(topicWordList, save_path)
        return topicWordList


    def get_x_only_Topics(self, dictProcess, ntop=10, cache_path=None):
        termMatrix = self.net.get_x_only_topics()
        #print(termMatrix.shape)
        topicWordList = []
        for each_topic in termMatrix:
            trans_list = list(enumerate(each_topic.cpu().numpy()))
            #print(trans_list)
            trans_list = sorted(trans_list, key=lambda k: k[1], reverse=True)
            #print(trans_list)
            topic_words = [dictProcess.get(item[0]) for item in trans_list[:ntop]]
            #print(topic_words)
            topicWordList.append(topic_words)
        if cache_path:
            save_path = os.path.join(cache_path, 'x_only_topics.txt')
            self.saveTopic(topicWordList, save_path)
        return topicWordList





    def saveTopic(self, topics, save_path):
        with open(save_path, 'w') as fo:
            for each_topic in topics:
                topic_line = ' '.join(each_topic)
                fo.write(topic_line+'\n')

















