import json
import glob
import os
import re
import hashlib
import logging
import sys

class DataReader:
    def __init__(self, annoed_json_dir, raw_json_dir, kwargs):

        self._read_input_options(kwargs)
        print('label ignored: ', self.ignoreLabelList)
        print('user ignored: ', self.ignoreUserList)
        self._setReaderLogger()

        self._anno_user_regex()
        self._read_raw_json(raw_json_dir)
        self._read_annoed_data(annoed_json_dir)

    def _setReaderLogger(self):
        self.readerLogger = logging.getLogger('readerLogger')
        if self.logging_level == 'info':
            self.readerLogger.setLevel(logging.INFO)
        elif self.logging_level == 'debug':
            self.readerLogger.setLevel(logging.DEBUG)
        elif self.logging_level == 'warning':
            self.readerLogger.setLevel(logging.WARNING)
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter('DataReader: %(message)s')
        handler.setFormatter(formatter)

        self.readerLogger.addHandler(handler)

    def logging(self, *args, logtype='info', sep=' '):
        getattr(self.readerLogger, logtype)(sep.join(str(a) for a in args))




    def _read_input_options(self, kwargs):
        self.ignoreLabelList = kwargs.get('ignoreLabelList', [])
        self.ignoreUserList = kwargs.get('ignoreUserList', [])
        self.confThres = kwargs.get('confThres', -1)
        self.filter_no_conf = kwargs.get('filter_no_conf', False)
        self.ignore_empty = kwargs.get('ignore_empty', False)
        self.label_trans_dict = kwargs.get('label_trans_dict', None)
        self.logging_level = kwargs.get('logging_level', 'warning')
        self.reverse_selection_condition = kwargs.get('reverse_selection_condition', False)
        self.annotator_level_confident = kwargs.get('annotator_level_confident', {})


    def _anno_user_regex(self):
        self.label_field_regex = re.compile('ann\d*\_label')
        self.annotator_id_regex = re.compile('(?<=ann)\d*(?=\_label)')
        self.confident_field_regex = re.compile('ann\d*\_conf')
        self.remark_field_regex = re.compile('ann\d*\_remarks')

    def _read_annoed_data(self, annoed_json_dir):
        all_jsons = glob.glob(annoed_json_dir+'/*.json')
        for each_annoed_json_file in all_jsons:
            self._read_annoed_json(each_annoed_json_file)
        #print(all_jsons)

    def _read_annoed_json(self, annoed_json):
        with open(annoed_json, 'r') as f_json:
            all_json_data = json.load(f_json)
        for each_annoed_data in all_json_data:
            uniqueIdentifier = self._get_unique_identifier(each_annoed_data)
            annotation_info, include = self._get_annotation_info(each_annoed_data)

            if len(annotation_info['label']) < 0 and ignore_empty:
                include = False

            if self.reverse_selection_condition:
                if include == False:
                    include = True
                else:
                    include = False

            if include:
                self.data_dict[uniqueIdentifier]['annotations'].append(annotation_info)

    def _translabel(self, label):
        if label in self.label_trans_dict:
            transferd_label = self.label_trans_dict[label]
        else:
            transferd_label = label
        return transferd_label
            
    def _get_annotation_info(self, dict_data):
        annotation_info_dict = {}
        dict_keys = dict_data.keys()
        include = True
        annotator_id = None
        for current_key in dict_keys:
            m_label = self.label_field_regex.match(current_key)
            if m_label:
                raw_label_field = m_label.group()
                #print(raw_label_field)
                annotator_id = self.annotator_id_regex.search(raw_label_field).group()
                #print(annotator_id)
                annotation_info_dict['annotator'] = annotator_id
                label = dict_data[raw_label_field]
                if self.label_trans_dict:
                    label = self._translabel(label)
                annotation_info_dict['label'] = label
                if label in self.ignoreLabelList:
                    include = False
                if annotator_id in self.ignoreUserList:
                    include = False
            m_conf = self.confident_field_regex.match(current_key)
            if m_conf:
                raw_conf_field = m_conf.group()
                confident = dict_data[raw_conf_field]
                annotation_info_dict['confident'] = confident
                if len(confident) < 1:
                    if self.filter_no_conf:
                        include = False
                elif annotator_id in self.annotator_level_confident:
                    if int(confident) <= self.annotator_level_confident[annotator_id]:
                        include = False
                elif (int(confident) <= self.confThres):
                    include = False

            m_remark = self.remark_field_regex.match(current_key)
            if m_remark:
                raw_remark_field = m_remark.group()
                remark = dict_data[raw_remark_field]
                #print(remark)
                annotation_info_dict['remark'] = remark

        return annotation_info_dict, include

                

    def _get_unique_identifier(self, each_data):
        source_link = each_data['Source'].strip()
        claim = each_data['Claim'].strip()
        explaination = each_data['Explaination'].strip()
        sourceToken = source_link.split('/')
        top3Source = ' '.join(sourceToken[:3])
        top200claim = claim[:200]
        top200expl = explaination[:200]
        uniqueString = top200claim+top200expl+top3Source

        #sourceQuesionToken = source_link.split('?')
        #uniqueString = sourceQuesionToken[0]
        #print(uniqueString)

        uniqueIdentifier = hashlib.sha224(uniqueString.encode('utf-8')).hexdigest()
        return uniqueIdentifier


    def _read_raw_json(self, raw_json_dir):
        self.data_dict = {}
        all_raw_jsons = glob.glob(raw_json_dir+'/*.json')
        duplicated = 0
        total_data = 0

        for each_raw_json in all_raw_jsons:
            ct=0
            cd=0
            with open(each_raw_json, 'r') as f_json:
                raw_data = json.load(f_json)
            for each_data in raw_data:
                #data_link = each_data['Link']
                uniqueIdentifier = self._get_unique_identifier(each_data)
                if uniqueIdentifier not in self.data_dict:
                    each_data['unique_wv_id'] = uniqueIdentifier
                    each_data['annotations'] = []
                    self.data_dict[uniqueIdentifier] = each_data
                else:
                    duplicated += 1
                    cd+=1
                    self.logging(uniqueIdentifier, logtype='debug')
                    self.logging('id: ', self.data_dict[uniqueIdentifier]['unique_wv_id'], logtype='debug')
                    self.logging(self.data_dict[uniqueIdentifier], logtype='debug')
                    self.logging(each_data, logtype='debug')
                    self.logging('\n', logtype='debug')
                total_data += 1
                ct+=1
            #print(ct, cd)
            #self.logging('Num selected data: ', len(self.data_dict), logtype='warning')

        self.logging('num duplicated: ', duplicated, logtype='info')
        self.logging('total num data: ', total_data, logtype='info')
