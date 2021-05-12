import nltk
from nltk.corpus import stopwords
import os
import re
from .PostprocessorBase import ReaderPostProcessorBase 
from transformers import BertTokenizer

class WVPostProcessor(ReaderPostProcessorBase):
    def __init__(self, x_fields=['Claim', 'Explaination'], y_field='selected_label', return_tokened=False, **kwargs):
        super().__init__(**kwargs)
        self.x_fields = x_fields
        self.y_field = y_field
        self.return_token = return_tokened
        self.toBow = True
        self.initProcessor()

    def initProcessor(self):
        bert_tokenizer_path = os.path.join(self.config['BERT'].get('bert_path'), 'tokenizer')
        print(bert_tokenizer_path)
        self.bert_tokenizer = BertTokenizer.from_pretrained(bert_tokenizer_path)
        self.tokenizerProcessor = self.bertTokenizer
        self.word2idProcessor = self.bertWord2id

        if 'TARGET' in self.config:
            self.labelsFields = self.config['TARGET'].get('labels')
        else:
            self.labelsFields = ['PubAuthAction', 'CommSpread', 'GenMedAdv', 'PromActs', 'Consp', 'VirTrans', 'VirOrgn', 'PubPrep', 'Vacc', 'Prot', 'None']
        #print(self.labelsFields)


    def postProcess(self, sample):
        split_x = []
        for x_field in self.x_fields:
            current_rawx = self._get_sample(sample, x_field)
            split_x.append(current_rawx)
        current_rawx = ' '.join(split_x)

        ## Bert toknise for hidden layers. add_special_tokens not added, additional attention will be applied on token level (CLS not used)
        if self.embd_ready:
            current_x = sample['embd']
        else:
            if self.return_token:
                current_x, tokened = self.x_pipeline(current_rawx, add_special_tokens=self.add_spec_tokens)
            else:
                current_x = self.x_pipeline(current_rawx, add_special_tokens=self.add_spec_tokens)
                tokened = None

        ## NLTK tokenise and remove stopwords for topic modelling
        current_x_nltk_tokened = self.nltkTokenizer(current_rawx)
        current_x_nltk_tokened = self._remove_stop_words(current_x_nltk_tokened)
        if self.dictProcess and self.toBow:
            current_x_nltk_tokened = self.dictProcess.doc2countHot(current_x_nltk_tokened)

        x=[current_x, current_x_nltk_tokened, tokened]

        y = sample[self.y_field]
        if self.label2id:
            y = self.label2ids(y)

        if self.remove_single_list:
            x = self._removeSingleList(x)
            y = self._removeSingleList(y)
        return x, y
