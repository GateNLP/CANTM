import sys
import nltk
import math
from GateMIcateLib import BatchIterBert, DictionaryProcess
#from GateMIcateLib import WVPostProcessor as ReaderPostProcessor
from configobj import ConfigObj
import torch
import argparse
import copy
from sklearn.model_selection import KFold
import random
import os
from pathlib import Path
from gensim.corpora.dictionary import Dictionary
from gensim.models import LdaModel
import numpy as np


def get_average_fmeasure_score(results_dict, field):
    t=0
    score = 0
    for class_field in results_dict['f-measure']:
        score += sum(results_dict['f-measure'][class_field][field])
        t += len(results_dict['f-measure'][class_field][field])
    return score/t

def get_micro_fmeasure(results_dict, num_field, de_field):
    score = 0
    for class_field in results_dict['f-measure']:
        numerator = sum(results_dict['f-measure'][class_field][num_field])
        denominator = sum(results_dict['f-measure'][class_field][de_field])
        if denominator != 0:
            score += numerator/denominator
    t = len(results_dict['f-measure'])

    return score/t



class EvaluationManager:
    def __init__(self, trainReaderargs, envargs, testReaderargs=None, valReaderargs=None):
        self._initParams(envargs)
        self.trainReaderargs = trainReaderargs
        self.testReaderargs = testReaderargs
        self.valReaderargs = valReaderargs
        self.getLibs()
        self._get_train_DataIter()

    def get_covid_train_json_for_scholar(self):
        current_traindataIter=dataIter(*self.trainReaderargs, config=self.config, shuffle=False)
        all_json = []
        
        for item in current_traindataIter:
            claim = item['Claim']
            explaination = item['Explaination']
            label = item['selected_label']
            sample_id = item['unique_wv_id']

            text = claim+' '+explaination

            current_dict = {}
            current_dict['text'] = text
            current_dict['sentiment'] = label
            current_dict['id'] = sample_id
            all_json.append(current_dict)
        return all_json


    def outputCorpus4NPMI(self):
        all_doc = []
        token_count = []
        current_traindataIter=dataIter(*self.trainReaderargs, config=self.config, shuffle=False)
        for item in current_traindataIter:
            alltext=[]
            for field in self.x_fields:
                current_text = nltk.word_tokenize(item[field])
                token_count.append(len(current_text))
                alltext.append(' '.join(current_text))
            all_doc.append(' '.join(alltext))
        if self.testReaderargs:
            self.testDataIter = dataIter(*self.testReaderargs, config=self.config, shuffle=False)
            for item in current_traindataIter:
                alltext=[]
                for field in self.x_fields:
                    current_text = nltk.word_tokenize(item[field])
                    token_count.append(len(current_text))
                    alltext.append(' '.join(current_text))
                all_doc.append(' '.join(alltext))
        print(sum(token_count)/len(token_count))
        return all_doc




    def _get_train_DataIter(self):
        self.postProcessor = ReaderPostProcessor(config=self.config, word2id=True, remove_single_list=False, add_spec_tokens=True, x_fields=self.x_fields, y_field=self.y_field, max_sent_len=self.max_sent_len)
        print(*self.trainReaderargs)
        self.trainDataIter = dataIter(*self.trainReaderargs, postProcessor=self.postProcessor, config=self.config, shuffle=True)

        if self.testReaderargs:
            self.testDataIter = dataIter(*self.testReaderargs, postProcessor=self.postProcessor, config=self.config, shuffle=False)

        print(self.get_dict)
        if self.get_dict:
            print('building dict')
            self.buildDict()
        if self.preEmbd:
            print('pre calculating embedding')
            net = Model(self.config, vocab_dim=self.vocab_dim)
            mUlti = modelUlti(net, gpu=self.gpu)
            self.trainDataIter.preCalculateEmbed(mUlti.net.bert_embedding, 0)

        if not self.testReaderargs:
            self.all_ids = copy.deepcopy(self.trainDataIter.all_ids)
            random.shuffle(self.all_ids)
            ## deep copy train reader to test reader
            self.testDataIter = copy.deepcopy(self.trainDataIter)
        self.valDataIter = None

    def _initParams(self,envargs):
        print(envargs)
        self.get_perp = False
        self.get_dict = False
        self.vocab_dim = None
        self.have_dict = False
        self.config_file = envargs.get('configFile',None)
        self.config = ConfigObj(self.config_file)
        self.cache_path = envargs.get('cachePath',None) 
        self.n_fold = envargs.get('nFold',5)
        self.randomSeed = envargs.get('randomSeed',None)
        self.preEmbd = envargs.get('preEmbd',False)
        self.dynamicSampling = envargs.get('dynamicSampling',False)
        self.modelType = envargs.get('model', 'clsTopic')
        self.corpusType = envargs.get('corpusType', 'wvmisinfo')
        self.max_sent_len = envargs.get('max_sent_len', '300')
        self.num_epoches = envargs.get('num_epoches', 150)
        self.patient = envargs.get('patient', 40)
        self.batch_size = envargs.get('batch_size', 32)
        self.earlyStopping = envargs.get('earlyStopping', 'cls_loss')
        self.x_fields = envargs.get('x_fields', 'Claim,Explaination')
        self.x_fields = self.x_fields.split(',')
        print(self.x_fields)
        self.y_field = envargs.get('y_field', 'selected_label')

        self.dict_no_below = envargs.get('dict_no_below', 0)
        self.dict_no_above = envargs.get('dict_no_above', 1.0)
        self.dict_keep_n = envargs.get('dict_keep_n', 5000)

        self.splitValidation = envargs.get('splitValidation',None)
        self.inspectTest = envargs.get('inspectTest', True)
        self.trainLDA = envargs.get('trainLDA', False)

        self.gpu = envargs.get('gpu', True)


        self.envargs = envargs


    def train_lda(self, cache_path):
        print(cache_path)
        trainBatchIter = BatchIterBert(self.trainDataIter, filling_last_batch=False, postProcessor=batchPostProcessor, batch_size=1)
        bow_list = []
        for item in trainBatchIter:
            bow = item[1].squeeze().detach().numpy().tolist()
            bow_list.append(self.bow_2_gensim(bow))
        print(len(bow_list))
        #print(self.dictProcess.common_dictionary.id2token)
        lda = LdaModel(np.array(bow_list), num_topics=50, passes=200, chunksize=len(bow_list), id2word=self.dictProcess.common_dictionary)
        #print(lda.show_topic(1, topn=10))
        output_topic_line = ''
        for topic_id in range(50):
            current_topic_list = []
            current_topic = lda.show_topic(topic_id, topn=10)
            for topic_tuple in current_topic:
                current_topic_list.append(topic_tuple[0])
            output_topic_line += ' '.join(current_topic_list)+'\n'
            #print(current_topic_list)

        topic_file = os.path.join(cache_path, 'ldatopic.txt')
        with open(topic_file, 'w') as fo:
            fo.write(output_topic_line)



            

        testBatchIter = BatchIterBert(self.testDataIter, filling_last_batch=False, postProcessor=batchPostProcessor, batch_size=1)

        test_bow_list = []
        word_count = 0
        for item in testBatchIter:
            bow = item[1].squeeze().detach().numpy().tolist()
            word_count += sum(bow)
            test_bow_list.append(self.bow_2_gensim(bow))


        
        print(word_count) 
        ppl = lda.log_perplexity(test_bow_list, len(test_bow_list))
        print(ppl)
        bound = lda.bound(test_bow_list)
        print(bound/word_count)
        print(np.exp2(-bound/word_count))

    def bow_2_gensim(self, bow):
        gensim_format = []
        for idx, count in enumerate(bow):
            if count > 0:
                gensim_format.append((idx,count))
        return gensim_format




    def train(self, cache_path=None):
        if self.inspectTest and (not self.splitValidation):
            print('inspecting test, please dont use val acc as early stoping')
            self.valDataIter = self.testDataIter
        elif self.inspectTest and self.splitValidation:
            print('inspectTest and splitValidation can not use same time')
            print('deset inspectTest')
            self.inspectTest = False

        if self.splitValidation:
            print('splitting test for validation')
            self.valDataIter = copy.deepcopy(self.trainDataIter)
            train_val_ids = copy.deepcopy(self.trainDataIter.all_ids)
            random.shuffle(train_val_ids)
            split_4_train = 1-self.splitValidation
            top_n_4_train = math.floor(len(train_val_ids) * split_4_train)
            id_4_train = train_val_ids[:top_n_4_train]
            id_4_val = train_val_ids[top_n_4_train:]
            self.trainDataIter.all_ids = id_4_train
            self.valDataIter.all_ids = id_4_val

        assert self.inspectTest != self.splitValidation, 'splitValidation will overwrite inspectTest, dont use at the same time'
        

        if self.dynamicSampling:
            print('get training data sample weights')
            trainDataIter.cal_sample_weights()

        self.trainDataIter._reset_iter()
        trainBatchIter = BatchIterBert(self.trainDataIter, filling_last_batch=True, postProcessor=batchPostProcessor, batch_size=self.batch_size)

        if self.valDataIter:
            self.valDataIter._reset_iter()
            valBatchIter = BatchIterBert(self.valDataIter, filling_last_batch=False, postProcessor=batchPostProcessor, batch_size=self.batch_size)
        else:
            valBatchIter = None

        print(self.vocab_dim)
        net = Model(self.config, vocab_dim=self.vocab_dim)
        self.mUlti = modelUlti(net, gpu=self.gpu)

        #print(next(trainBatchIter))

        self.mUlti.train(trainBatchIter, cache_path=cache_path, num_epohs=self.num_epoches, valBatchIter=valBatchIter, patience=self.patient, earlyStopping=self.earlyStopping)


    def train_test_evaluation(self):
        path = Path(self.cache_path)
        path.mkdir(parents=True, exist_ok=True)
        self.train(cache_path=self.cache_path)
        testBatchIter = BatchIterBert(self.testDataIter, filling_last_batch=False, postProcessor=batchPostProcessor, batch_size=self.batch_size)
        results = self.mUlti.eval(testBatchIter, get_perp=self.get_perp)
        print(results)

    def train_model_only(self):
        path = Path(self.cache_path)
        path.mkdir(parents=True, exist_ok=True)
        self.train(cache_path=self.cache_path)

    def cross_fold_evaluation(self):
        kf = KFold(n_splits=self.n_fold)
        fold_index = 1
        results_dict = {}
        results_dict['accuracy'] = []
        results_dict['perplexity'] = []
        results_dict['log_perplexity'] = []
        results_dict['perplexity_x_only'] = []
        results_dict['f-measure'] = {}
        for each_fold in kf.split(self.all_ids):
            train_ids, test_ids = self.reconstruct_ids(each_fold)
            self.trainDataIter.all_ids = train_ids
            self.testDataIter.all_ids = test_ids
            self.testDataIter._reset_iter()
            fold_cache_path = os.path.join(self.cache_path, 'fold'+str(fold_index))
            path = Path(fold_cache_path)
            path.mkdir(parents=True, exist_ok=True)

            if self.trainLDA:
                self.train_lda(cache_path=fold_cache_path)
            else:
                self.train(cache_path=fold_cache_path)

                testBatchIter = BatchIterBert(self.testDataIter, filling_last_batch=False, postProcessor=batchPostProcessor, batch_size=self.batch_size)

                results = self.mUlti.eval(testBatchIter, get_perp=self.get_perp)
                print(results)
                results_dict['accuracy'].append(results['accuracy'])
                if 'perplexity' in results:
                    results_dict['perplexity'].append(results['perplexity'])
                    results_dict['log_perplexity'].append(results['log_perplexity'])
                    results_dict['perplexity_x_only'].append(results['perplexity_x_only'])

                for f_measure_class in results['f-measure']:
                    if f_measure_class not in results_dict['f-measure']:
                        results_dict['f-measure'][f_measure_class] = {'precision':[], 'recall':[], 'f-measure':[], 'total_pred':[], 'total_true':[], 'matches':[]}
                    results_dict['f-measure'][f_measure_class]['precision'].append(results['f-measure'][f_measure_class][0])
                    results_dict['f-measure'][f_measure_class]['recall'].append(results['f-measure'][f_measure_class][1])
                    results_dict['f-measure'][f_measure_class]['f-measure'].append(results['f-measure'][f_measure_class][2])
                    results_dict['f-measure'][f_measure_class]['total_pred'].append(results['f-measure'][f_measure_class][3])
                    results_dict['f-measure'][f_measure_class]['total_true'].append(results['f-measure'][f_measure_class][4])
                    results_dict['f-measure'][f_measure_class]['matches'].append(results['f-measure'][f_measure_class][5])
            fold_index += 1

        print(results_dict)
        overall_accuracy = sum(results_dict['accuracy'])/len(results_dict['accuracy'])
        if len(results_dict['perplexity']) >0:
            overall_perplexity = sum(results_dict['perplexity'])/len(results_dict['perplexity'])
            print('perplexity: ', overall_perplexity)
            overall_log_perplexity = sum(results_dict['log_perplexity'])/len(results_dict['log_perplexity'])
            print('log perplexity: ', overall_log_perplexity)
            overall_perplexity_x = sum(results_dict['perplexity_x_only'])/len(results_dict['perplexity_x_only'])
            print('perplexity_x_only: ', overall_perplexity_x)



        macro_precision = get_average_fmeasure_score(results_dict, 'precision')
        macro_recall = get_average_fmeasure_score(results_dict, 'recall')
        macro_fmeasure = get_average_fmeasure_score(results_dict, 'f-measure')

        micro_precision = get_micro_fmeasure(results_dict, 'matches', 'total_pred')
        micro_recall = get_micro_fmeasure(results_dict, 'matches', 'total_true')
        micro_fmeasure = 2*((micro_precision*micro_recall)/(micro_precision+micro_recall))

        print('accuracy: ', overall_accuracy)
        print('micro_precision: ', micro_precision)
        print('micro_recall: ', micro_recall)
        print('micro_f-measure: ', micro_fmeasure)
        print('macro_precision: ', macro_precision)
        print('macro_recall: ', macro_recall)
        print('macro_f-measure: ', macro_fmeasure)

    def reconstruct_ids(self, each_fold):
        output_ids = [[],[]] #[train_ids, test_ids]
        for sp_id in range(len(each_fold)):
            current_output_ids = output_ids[sp_id]
            current_fold_ids = each_fold[sp_id]
            for doc_id in current_fold_ids:
                current_output_ids.append(self.all_ids[doc_id])
        return output_ids

    def buildDict(self):
        batchiter = BatchIterBert(self.trainDataIter, filling_last_batch=False, postProcessor=xonlyBatchProcessor, batch_size=1)
        common_dictionary = Dictionary(batchiter)
        print(len(common_dictionary))
        if self.testReaderargs:
            print('update vocab from test set')
            batchiter = BatchIterBert(self.testDataIter, filling_last_batch=False, postProcessor=xonlyBatchProcessor, batch_size=1)
            common_dictionary.add_documents(batchiter)
            print(len(common_dictionary))
            
        common_dictionary.filter_extremes(no_below=self.dict_no_below, no_above=self.dict_no_above, keep_n=self.dict_keep_n)
        self.dictProcess = DictionaryProcess(common_dictionary)
        self.postProcessor.dictProcess = self.dictProcess
        self.vocab_dim = len(self.dictProcess)
        self.have_dict = True

        if 1:
            count_list = []
            self.trainDataIter._reset_iter()
            batchiter = BatchIterBert(self.trainDataIter, filling_last_batch=False, postProcessor=xonlyBatchProcessor, batch_size=1)
            for item in batchiter:
                current_count = sum(item)
                count_list.append(current_count)
                #print(current_count)
            print(sum(count_list)/len(count_list))



    def getModel(self):
        self.net = Model(config, vocab_dim=vocab_dim)


    def getLibs(self):
        print('getting libs')
        print(self.modelType)

        global modelUlti
        global Model
        global xonlyBatchProcessor
        global batchPostProcessor
        global dataIter
        global ReaderPostProcessor

        if self.modelType == 'clsTopic':
            from GateMIcateLib import ModelUltiClass as modelUlti
            from GateMIcateLib.models import CLSAW_TopicModel as Model
            from GateMIcateLib.batchPostProcessors import xonlyBatchProcessor
            from GateMIcateLib.batchPostProcessors import bowBertBatchProcessor as batchPostProcessor
            self.get_dict = True
            self.get_perp = True

        elif self.modelType == 'clsTopicSL':
            from GateMIcateLib import ModelUltiClass as modelUlti
            from GateMIcateLib.models import CLSAW_TopicModelSL as Model
            from GateMIcateLib.batchPostProcessors import xonlyBatchProcessor
            from GateMIcateLib.batchPostProcessors import bowBertBatchProcessor as batchPostProcessor
            self.get_dict = True
            self.get_perp = True

        elif self.modelType == 'baselineBert':
            from GateMIcateLib import ModelUlti as modelUlti
            from GateMIcateLib.models import BERT_Simple as Model
            from GateMIcateLib.batchPostProcessors import xyOnlyBertBatchProcessor as batchPostProcessor

        elif self.modelType == 'nvdm':
            from GateMIcateLib import ModelUltiClass as modelUlti
            from GateMIcateLib.models import NVDM as Model
            from GateMIcateLib.batchPostProcessors import xonlyBatchProcessor
            from GateMIcateLib.batchPostProcessors import bowBertBatchProcessor as batchPostProcessor
            self.get_dict = True
            self.get_perp = True

        elif self.modelType == 'orinvdm':
            from GateMIcateLib import ModelUltiClass as modelUlti
            from GateMIcateLib.models import ORINVDM as Model
            from GateMIcateLib.batchPostProcessors import xonlyBatchProcessor
            from GateMIcateLib.batchPostProcessors import bowBertBatchProcessor as batchPostProcessor
            self.get_dict = True
            self.get_perp = True

        elif self.modelType == 'clsTopicBE':
            from GateMIcateLib import ModelUltiClass as modelUlti
            from GateMIcateLib.models import CLSAW_TopicModel_BERTEN as Model
            from GateMIcateLib.batchPostProcessors import xonlyBatchProcessor
            from GateMIcateLib.batchPostProcessors import bowBertBatchProcessor as batchPostProcessor
            self.get_dict = True
            self.get_perp = True



        print(self.corpusType)
        if self.corpusType == 'wvmisinfo':
            from GateMIcateLib.readers import WVmisInfoDataIter as dataIter
            from GateMIcateLib import WVPostProcessor as ReaderPostProcessor
            self.dict_no_below = 3
            self.dict_no_above = 0.7
        elif self.corpusType == 'wvmisinfoScholar':
            from GateMIcateLib.readers import WVmisInfoDataIter as dataIter
            from GateMIcateLib import ScholarPostProcessor as ReaderPostProcessor
            self.dict_keep_n = 2000
        elif self.corpusType == 'aclIMDB':
            from GateMIcateLib.readers import ACLimdbReader as dataIter
            from GateMIcateLib import ScholarPostProcessor as ReaderPostProcessor
        elif self.corpusType == 'tsvBinary':
            from GateMIcateLib.readers import TsvBinaryFolderReader as dataIter
            from GateMIcateLib import WVPostProcessor as ReaderPostProcessor
            self.dict_no_below = 3
            self.dict_no_above = 0.7







