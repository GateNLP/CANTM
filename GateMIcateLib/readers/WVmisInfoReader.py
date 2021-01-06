import random
import math
import json
import torch
from .ReaderBase import CLSReaderBase

class WVmisInfoDataIter(CLSReaderBase):
    def __init__(self, merged_json, label_field='category', **kwargs):
        super().__init__(**kwargs)
        self.label_field = label_field

        self._initReader(merged_json)
        self._reset_iter()

    def _initReader(self, merged_json):
        with open(merged_json, 'r') as f_json:
            merged_data = json.load(f_json)
        self.all_ids = []
        self.data_dict = {}
        
        numberid = 0

        for item in merged_data:
            select = True
            annotation = item[self.label_field]
            if self.target_labels:
                if annotation not in self.target_labels:
                    #self.all_ids.append(item['unique_wv_id'])
                    #self.data_dict[item['unique_wv_id']] = item
                    select = False

            if select:
                try:
                    self.all_ids.append(item['unique_wv_id'])
                    self.data_dict[item['unique_wv_id']] = item
                except:
                    self.all_ids.append(str(numberid))
                    self.data_dict[str(numberid)] = item
            
            numberid += 1
