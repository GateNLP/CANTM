import nltk
from nltk.corpus import stopwords
import os
import re

class ReaderPostProcessorBase:
    def __init__(self, 
            keep_case=False, 
            label2id=True, 
            config=None,
            word2id=False,
            exteralWord2idFunction=None,
            return_mask=False,
            remove_single_list=True,
            max_sent_len = 300,
            add_spec_tokens = False,
            add_CLS_token = False,
            ):
        self._init_defaults()
        self.add_CLS_token = add_CLS_token
        self.add_spec_tokens = add_spec_tokens
        self.max_sent_len = max_sent_len
        self.keep_case = keep_case
        self.label2id = label2id
        self.word2id = word2id
        self.exteralWord2idFunction = exteralWord2idFunction
        self.config = config
        self.return_mask = return_mask
        self.remove_single_list = remove_single_list

    def _init_defaults(self):
        self.labelsFields = None
        self.stop_words = set(stopwords.words('english'))
        self.dictProcess = None
        self.embd_ready = False

    def _remove_stop_words(self, tokened):
        remain_list = []
        for token in tokened:
            contain_symbol, contain_number, contain_char, all_asii = self._check_string(token)
            keep = True
            if token in self.stop_words:
                keep = False
            elif len(token) == 1:
                keep = False
            elif token.isdigit():
                keep = False
            elif not contain_char:
                keep = False
            elif not all_asii:
                keep = False
            elif len(token) > 18:
                keep = False

            if keep == True:
                remain_list.append(token)
            else:
                pass
                #print(token)
        return remain_list

    def _check_string(self, inputString):
        contain_symbol = False
        contain_number = False
        contain_char = False
        all_asii = True


        have_symbol = re.compile('[@_!#$%^&*()<>?/\|}{~:]')
        have_number = re.compile('\d')
        have_char = re.compile('[a-zA-Z]')

        ms = have_symbol.search(inputString)
        if ms:
            contain_symbol = True

        mn = have_number.search(inputString)
        if mn:
            contain_number = True

        mc = have_char.search(inputString)
        if mc:
            contain_char = True

        if contain_char and not contain_number and not contain_symbol:
            all_asii = all(ord(c) < 128 for c in inputString)


        return contain_symbol, contain_number, contain_char, all_asii


    def _removeSingleList(self, y):
        if len(y) == 1:
            return y[0]
        else:
            return y

    def _get_sample(self, sample, sample_field):
        current_rawx = sample[sample_field]
        if self.keep_case == False:
            current_rawx = current_rawx.lower()
        return current_rawx


    def label2ids(self, label):
        label_index = self.labelsFields.index(label)
        return label_index


    def x_pipeline(self, raw_x, add_special_tokens=True):
        raw_x = self.tokenizerProcessor(raw_x)
        if self.word2id:
            raw_x = self.word2idProcessor(raw_x, add_special_tokens=add_special_tokens)
        return raw_x

    def nltkTokenizer(self, text):
        return nltk.word_tokenize(text)

    def bertTokenizer(self, text):
        tokened = self.bert_tokenizer.tokenize(text)
        #print(tokened)
        #ided = self.bert_tokenizer.encode_plus(tokened, max_length=100, pad_to_max_length=True, is_pretokenized=True, add_special_tokens=True)['input_ids']
        #print(ided)
        return tokened

    def bertWord2id(self,tokened, add_special_tokens=True):
        encoded = self.bert_tokenizer.encode_plus(tokened, max_length=self.max_sent_len, pad_to_max_length=True, is_pretokenized=True, add_special_tokens=add_special_tokens)
        #print(encoded)
        ided = encoded['input_ids']
        if self.return_mask:
            mask = encoded['attention_mask']
            return ided, mask
        else:
            return ided

    def get_label_desc_ids(self):
        label_desc_list = []
        for label in self.labelsFields:
            label_desc = self.desctiptionDict[label]
            current_desc_ids = self.x_pipeline(label_desc, max_length=100)
            label_desc_list.append(current_desc_ids)
        label_ids = [s[0] for s in label_desc_list]
        label_mask_ids = [s[1] for s in label_desc_list]
        return label_ids, label_mask_ids







