"""Simple example to do very simple whitespace-tokenization"""

import sys
import re
from gatenlp import interact, GateNlpPr, Document
from GateMIcateLib import ModelUltiClass as ModelUlti
import torch
from GateMIcateLib import WVPostProcessor as ReaderPostProcessor
from configobj import ConfigObj
from GateMIcateLib.batchPostProcessors import singleProcessor_noy_att as batchPostProcessor
import torch.nn.functional as F
import os
import copy
import string
from GateMIcateLib.models import CLSAW_TopicModel as Model

def reconstruct_word_attention(token_list, attention_weights, debugging = False):
    recon_token_list, normalise_att_weights = single_att_reconstruction(token_list, attention_weights)
    #print(recon_token_list, normalise_att_weights)
    #topn = torch.topk(normalise_att_weights, 5)
    #print(topn)
    #topn_indices = topn.indices
    #topn_values = topn.values

    topn_values, topn_indices = torch.sort(normalise_att_weights, descending=True)
    if debugging: print(topn_values)
    #print(top3)
    topn_list = []
    thres=0.4
    t=0
    for i in range(len(topn_values)):
        t+= topn_values[i]
        if debugging: print(t)
        if t.item() >= thres:
            cut_point = i+1
            break
    topn_values = topn_values[:cut_point]
    topn_indices = topn_indices[:cut_point]


    return topn_indices.to('cpu').detach().numpy(), topn_values.to('cpu').detach().numpy(), recon_token_list

def single_att_reconstruction(single_token_list, attention_weights):
    #print(single_token_list)
    bert_hash_token = re.compile('##.*')
    recon_token_list = []
    recon_att_weight_list = []
    attention_weights = attention_weights[1:]
    #print(single_token_list)
    for i in range(len(single_token_list)):
        current_token = single_token_list[i]
        current_att_weight = attention_weights[i]
        m = bert_hash_token.match(current_token)
        if m:
            current_reconst_token = current_token[2:]
            recon_token_list[-1] += current_reconst_token
            recon_att_weight_list[-1].append(current_att_weight)
        else:
            recon_token_list.append(current_token)
            recon_att_weight_list.append([current_att_weight])

    recon_att_weight_list_new = []
    for item in recon_att_weight_list:
        if len(item) > 1:
            recon_att_weight_list_new.append(sum(item)/len(item))
        else:
            recon_att_weight_list_new.append(item[0])

    normalise_att_weights = torch.nn.functional.softmax(torch.tensor(recon_att_weight_list_new))
    return recon_token_list, normalise_att_weights



@GateNlpPr
class MyProcessor:
    def __init__(self):
        script_path = os.path.abspath(__file__)
        print(script_path)
        parent = os.path.dirname(script_path)
        print(parent)
        stop_list_dir = os.path.join(parent, 'GateMIcateLib')
        stop_list_dir = os.path.join(stop_list_dir, 'stopwords')
        snowball_stopwords_list_file = os.path.join(stop_list_dir, 'snowball_stopwords.txt')
        self.stop_words = set(string.punctuation)
        with open(snowball_stopwords_list_file, 'r') as fin:
            for line in fin:
                stop_word = line.strip()
                self.stop_words.add(stop_word)
        print(self.stop_words)

        model_path = os.path.join(parent, 'FirstDraftModel2')
        config_path = os.path.join(parent, 'vaccine_gateapp.config')
        self.config = ConfigObj(config_path)
        #self.config['BERT'] = {}
        #self.config['BERT']['bert_path'] = os.path.join(parent, 'bert-base-uncased')
        gpu = self.config.get('MODEL').as_bool('gpu')
        print(gpu)

        #net = Model(self.config, vocab_dim=2000)
        #net.load_state_dict(torch.load(os.path.join(model_path,'bestnet.weights'), map_location=torch.device('cpu')), strict=False)

        self.mUlti = ModelUlti(load_path=model_path, gpu=gpu)
        #self.mUlti.net = net
        #self.mUlti.saveWeights(os.path.join(model_path,'bestnet.weights'))
        #self.mUlti.saveModel(os.path.join(parent, 'FirstDraftModel2'))

        self.x_fields = ['text']
        self.y_field = 'label'
        self.postProcessor = ReaderPostProcessor(config=self.config, return_tokened=True, word2id=True, remove_single_list=False, add_spec_tokens=True, x_fields=self.x_fields, y_field=self.y_field, label2id=False)
        self.postProcessor.dictProcess = self.mUlti.bowdict
        self.label_list = self.mUlti.labels
        self.topicList = self.mUlti.getTopics(self.mUlti.bowdict)
        self.tokens_total = 0
        self.nr_docs = 0

    def start(self, **kwargs):
        #print('start')
        self.nr_docs = 0
        self.nr_docs = 0

    def finish(self, **kwargs):
        print("Total number of tokens:", self.tokens_total)
        print("Number of documents:", self.nr_docs)

    def __call__(self, doc, **kwargs):
        debugging = (kwargs.get("debug") != 'no')

        set1 = doc.annset(kwargs.get("outputASName", "VaccineCate"))
        set1.clear()
        text = doc.text
        text = text.lower()
        if debugging: print(text)
        start_off = 0
        end_off = len(text)

        sample = {}
        sample['text'] = text
        sample['label'] = None
        postprocessed = self.postProcessor.postProcess(sample)
        #print(postprocessed[0])
        single_x, idded_words, _, tokened = batchPostProcessor(postprocessed[0])
        current_prediction = self.mUlti.application_oneSent(single_x, idded_words)
        #print(current_prediction)
        pred = current_prediction['pred']['y_hat']
        cls_att = current_prediction['pred']['cls_att']
        cls_att = cls_att.squeeze(0).to('cpu').detach().numpy()
        #print(cls_att.shape)
        #print(tokened)
        topn_indices, topn_values, recon_token_list = reconstruct_word_attention(tokened[0], cls_att, debugging)
        #print(topn_list)
        if debugging: print(recon_token_list)


        atted_topics = current_prediction['pred']['topics']
        top_topics = torch.topk(atted_topics, k=3, dim=1)
        topic_idx = top_topics[1]
        topic_score = top_topics[0]
        if debugging: print(topic_idx)
        current_batch_out = F.softmax(pred, dim=-1)
        #print('batchout: ', current_batch_out)
        sortedout, indicesout = torch.sort(current_batch_out)
        #print(sortedout, indicesout)
        #print(torch.max(current_batch_out, -1))
        label_prediction = torch.max(current_batch_out, -1)[1]
        #print(indicesout[0])
        label_prediction_2nd = indicesout[0][-2].to('cpu').detach().numpy()
        #print(label_prediction_2nd)
        label_prediction_2nd_score = sortedout[0][-2].to('cpu').detach().numpy()
        #print(1111,label_prediction_2nd_score)
        label_prediction_score = sortedout[0][-1].to('cpu').detach().numpy()

        current_label_ids = label_prediction.to('cpu').detach().numpy()[0]
        trans_label = self.label_list[current_label_ids]
        trans_label_2nd = self.label_list[label_prediction_2nd]

        feature_map = {}
        feature_map["class"] = trans_label
        feature_map["score"] = str(label_prediction_score)

        feature_map["2nd_class"] = trans_label_2nd
        feature_map["2nd_score"] = str(label_prediction_2nd_score)




        i=1
        for current_top_topics in topic_idx[0]:
            if debugging: print(current_top_topics)
            topic_id = current_top_topics.cpu().item()
            if debugging: print(topic_id)
            current_topic_wods = self.topicList[topic_id]
            if debugging: print(current_topic_wods)
            current_feature_name = 'top'+str(i)+' topic'
            feature_map[current_feature_name] = ','.join(current_topic_wods)
            i+=1

        set1.add(start_off, end_off, kwargs.get("misinfoAnnotationType", "VaccineClass"), feature_map)
        output_feature_map = {}
        output_feature_map["score"] = str(label_prediction_score)
        set1.add(start_off, end_off, trans_label, output_feature_map)



        #set2 = doc.get_annotations(kwargs.get("outputASName", "MisInformation"))
        #set2.clear()
        token_id = 0
        off_set_id = 0
        stored_off_set = 0
        found=False
        token_anno_list = []
        while token_id != (len(recon_token_list)):
            current_token = recon_token_list[token_id]
            current_string = ''
            for current_off_set in range(off_set_id, len(text)):
                current_char = text[current_off_set]
                current_string += current_char
                if current_string == current_token:
                    token_feature_map = {}
                    token_feature_map['token_id'] = token_id
                    token_feature_map['start_off_set'] = off_set_id
                    token_feature_map['end_off_set'] = current_off_set+1
                    token_feature_map['string'] = current_string

                    set1.add(off_set_id, current_off_set+1, kwargs.get("annotationType", "Token"), token_feature_map)
                    deepcopyStart = copy.deepcopy(off_set_id)
                    deepcopyEnd = copy.deepcopy(current_off_set+1)
                    token_anno_list.append([deepcopyStart, deepcopyEnd])
                    off_set_id = current_off_set+1
                    token_id += 1
                    found = True
                    #print(current_string)
                    stored_off_set = copy.deepcopy(off_set_id)
                    break

            if found:
                found = False
            else:
                off_set_id += 1

            if off_set_id > len(text):
                token_id += 1
                off_set_id = copy.deepcopy(stored_off_set)

        if debugging: print(recon_token_list)
        for current_top_idx in range(len(topn_indices)):
            current_top_index = topn_indices[current_top_idx]
            current_top_value = topn_values[current_top_idx]
            current_top_string = recon_token_list[current_top_index]
            if current_top_string not in self.stop_words:
                if debugging: print(current_top_string)
                fm = {}
                fm['score'] = str(current_top_value)
                fm['string'] = str(current_top_string)
                if debugging: print(fm)
                start_off_set = token_anno_list[current_top_index][0]
                end_off_set = token_anno_list[current_top_index][1]
                set1.add(start_off_set, end_off_set, kwargs.get("attentionAnnotationType", "Attention"), fm)


if __name__ == '__main__':
    interact()
