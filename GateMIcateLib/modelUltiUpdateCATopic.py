import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import copy
import os
from pathlib import Path
import pickle
from .modelUltiClassTopic import ModelUltiClass

class ModelUltiUpdateCAtopic(ModelUltiClass):
    def __init__(self, net=None, gpu=False, load_path=None):
        super().__init__(net=net, gpu=gpu, load_path=load_path)


    def train(self, trainBatchIter, num_epohs=100, valBatchIter=None, cache_path=None, earlyStopping='cls_loss', patience=5):
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
            all_loss = []
            all_elboz1 = []
            all_elboz2 = []
            all_bow = []
            trainIter = self.pred(trainBatchIter, train=False, updateTopic=True)
            for current_prediction in trainIter:
                self.optimizer.zero_grad()

                pred = current_prediction['pred']
                atted = current_prediction['atted']
                loss = pred['loss']
                bow_x = current_prediction['x_bow'].to('cpu').detach().numpy()
                all_bow.append(bow_x)
                loss.backward()
                self.optimizer.step()
                topics = self.getTopics(trainBatchIter.dataIter.postProcessor.dictProcess, cache_path=self.cache_path)
            topics = self.getTopics(trainBatchIter.dataIter.postProcessor.dictProcess, cache_path=self.cache_path)
            cache_last_path = os.path.join(self.cache_path, 'last_net.model')
            self.saveWeights(cache_last_path)

            print('finish epoch ', epoch)
        cache_last_path = os.path.join(self.cache_path, 'last_net.model')
        self.saveWeights(cache_last_path)
        self.saveModel(self.cache_path)
        self.getTopics(trainBatchIter.dataIter.postProcessor.dictProcess, cache_path=self.cache_path)
















