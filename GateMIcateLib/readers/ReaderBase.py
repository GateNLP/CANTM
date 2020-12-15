import random
import math
import json
import torch

class CLSReaderBase:
    def __init__(self, postProcessor=None, shuffle=False, config=None):
        self.label_count_dict = {}
        self.label_weights_list = None
        self._readConfigs(config)
        self.shuffle = shuffle
        self.postProcessor = postProcessor
        self.goPoseprocessor = True

    def _readConfigs(self, config):
        self.target_labels = None
        if config:
            if 'TARGET' in config:
                self.target_labels = config['TARGET'].get('labels')
                print(self.target_labels)

    def __iter__(self):
        if self.shuffle:
            random.shuffle(self.all_ids)
        self._reset_iter()
        return self

    def __next__(self):
        #print(self.all_ids)
        if self.current_sample_idx < len(self.all_ids):
            current_sample = self._readNextSample()
            self.current_sample_idx += 1
            return current_sample

        else:
            self._reset_iter()
            raise StopIteration


    def _readNextSample(self):
        current_id = self.all_ids[self.current_sample_idx]
        #print(current_id)
        self.current_sample_dict_id = current_id
        current_sample = self.data_dict[current_id]
        if self.postProcessor and self.goPoseprocessor:
            current_sample = self.postProcessor.postProcess(current_sample)
        return current_sample

    def preCalculateEmbed(self, embd_net, embd_field, dataType=torch.long, device='cuda:0'):
        for sample, _ in self:
            x_embd = sample[embd_field]
            input_tensor = torch.tensor([x_embd], dtype=torch.long, device=device)
            with torch.no_grad():
                embd = embd_net(input_tensor)
            self.data_dict[self.current_sample_dict_id]['embd'] = embd[0].tolist()

        self.postProcessor.embd_ready = True

        #pass


    def __len__(self):
        return len(self.all_ids)

    def _reset_iter(self):
        if self.shuffle:
            random.shuffle(self.all_ids)
        self.current_sample_idx = 0
        #print(self.all_ids)
        self.current_sample_dict_id = self.all_ids[self.current_sample_idx]

    def count_samples(self):
        self.goPoseprocessor = False
        self.label_count_dict = {}
        self.label_count_list = [0]*len(self.postProcessor.labelsFields)
        for item in self:
            #print(item)
            annotation = item['selected_label']
            annotation_idx = self.postProcessor.labelsFields.index(annotation)
            self.label_count_list[annotation_idx] += 1
            if annotation not in self.label_count_dict:
                self.label_count_dict[annotation] = 0
            self.label_count_dict[annotation] += 1
        print(self.label_count_dict)
        print(self.label_count_list)
        self.goPoseprocessor = True

    def cal_sample_weights(self):
        self.count_samples()
        self.label_weights_list = []
        max_count = max(self.label_count_list)
        for i in range(len(self.label_count_list)):
            current_count = self.label_count_list[i]
            num_samples = math.ceil(max_count/current_count)
            self.label_weights_list.append(num_samples)
        print(self.label_weights_list)




