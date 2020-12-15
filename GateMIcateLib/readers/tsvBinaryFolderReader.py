import random
import math
import json
import torch
import glob
import os
from .ReaderBase import CLSReaderBase

class TsvBinaryFolderReader(CLSReaderBase):
    def __init__(self, input_dir, pos_folder='positive', neg_folder='negative', text_filed=1, id_field=0,  **kwargs):
        super().__init__(**kwargs)
        self.text_filed = text_filed
        self.id_field = id_field
        pos_dir = os.path.join(input_dir, pos_folder)
        neg_dir = os.path.join(input_dir, neg_folder)
        self._initReader(pos_dir, neg_dir)
        self._reset_iter()

    def _initReader(self, pos_dir, neg_dir):
        self.all_ids = []
        self.data_dict = {}
        self.global_ids = 0
        all_pos_file_list = glob.glob(pos_dir+'/*.tsv')
        self._readDir(all_pos_file_list, 'pos')

        all_neg_file_list = glob.glob(neg_dir+'/*.tsv')
        self._readDir(all_neg_file_list, 'neg')

    def _readDir(self, file_list, label):
        for each_file in file_list:
            self._readFile(each_file, label)

    def _readFile(self, current_file, label):
        current_text_id = str(self.global_ids)
        with open(current_file, 'r') as fin:
            for line in fin:
                lineTok = line.split('\t')
                #print(self.id_field)
                if self.id_field != None:
                    #print('ssssssssssssss')
                    current_text_id = lineTok[self.id_field]
                raw_text = lineTok[self.text_filed]
                #print(current_text_id)
                if current_text_id not in self.data_dict:
                    self.data_dict[current_text_id] = {}
                    self.data_dict[current_text_id]['text'] = raw_text
                    self.data_dict[current_text_id]['selected_label'] = label
                    self.all_ids.append(current_text_id)
                else:
                    self.data_dict[current_text_id]['text'] += '\n'+raw_text
        self.global_ids += 1



