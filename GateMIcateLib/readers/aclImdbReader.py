import random
import math
import json
import torch
import glob
import os
from .ReaderBase import CLSReaderBase

class ACLimdbReader(CLSReaderBase):
    def __init__(self, input_dir, **kwargs):
        super().__init__(**kwargs)
        pos_dir = os.path.join(input_dir,'pos')
        neg_dir = os.path.join(input_dir,'neg')
        self._initReader(pos_dir, neg_dir)
        self._reset_iter()

    def _initReader(self, pos_dir, neg_dir):
        self.all_ids = []
        self.data_dict = {}
        self.global_ids = 0
        all_pos_file_list = glob.glob(pos_dir+'/*.txt')
        self._readDir(all_pos_file_list, 'pos')

        all_neg_file_list = glob.glob(neg_dir+'/*.txt')
        self._readDir(all_neg_file_list, 'neg')

    def _readDir(self, file_list, label):
        for each_file in file_list:
            self._readFile(each_file, label)

    def _readFile(self, current_file, label):
        current_text_id = str(self.global_ids)
        with open(current_file, 'r') as fin:
            all_text = fin.readlines()
            raw_text = ' '.join(all_text)
            self.data_dict[current_text_id] = {}
            self.data_dict[current_text_id]['text'] = raw_text
            self.data_dict[current_text_id]['selected_label'] = label
            self.all_ids.append(current_text_id)
        self.global_ids += 1



