import nltk
from nltk.corpus import stopwords
import os
import re
from .PostprocessorBase import ReaderPostProcessorBase 
from transformers import BertTokenizer

def tokenize(text, strip_html=False, lower=True, keep_emails=False, keep_at_mentions=False, keep_numbers=False, keep_alphanum=False, min_length=3, stopwords=None, vocab=None):
    text = clean_text(text, strip_html, lower, keep_emails, keep_at_mentions)
    tokens = text.split()

    if stopwords is not None:
        tokens = ['_' if t in stopwords else t for t in tokens]

    # remove tokens that contain numbers
    if not keep_alphanum and not keep_numbers:
        tokens = [t if alpha.match(t) else '_' for t in tokens]

    # or just remove tokens that contain a combination of letters and numbers
    elif not keep_alphanum:
        tokens = [t if alpha_or_num.match(t) else '_' for t in tokens]

    # drop short tokens
    if min_length > 0:
        tokens = [t if len(t) >= min_length else '_' for t in tokens]

    counts = Counter()

    unigrams = [t for t in tokens if t != '_']
    counts.update(unigrams)

    if vocab is not None:
        tokens = [token for token in unigrams if token in vocab]
    else:
        tokens = unigrams

    return tokens, counts


def clean_text(text, strip_html=False, lower=True, keep_emails=False, keep_at_mentions=False):
    # remove html tags
    if strip_html:
        text = re.sub(r'<[^>]+>', '', text)
    else:
        # replace angle brackets
        text = re.sub(r'<', '(', text)
        text = re.sub(r'>', ')', text)
    # lower case
    if lower:
        text = text.lower()
    # eliminate email addresses
    if not keep_emails:
        text = re.sub(r'\S+@\S+', ' ', text)
    # eliminate @mentions
    if not keep_at_mentions:
        text = re.sub(r'\s@\S+', ' ', text)
    # replace underscores with spaces
    text = re.sub(r'_', ' ', text)
    # break off single quotes at the ends of words
    text = re.sub(r'\s\'', ' ', text)
    text = re.sub(r'\'\s', ' ', text)
    # remove periods
    text = re.sub(r'\.', '', text)
    # replace all other punctuation (except single quotes) with spaces
    text = replace.sub(' ', text)
    # remove single quotes
    text = re.sub(r'\'', '', text)
    # replace all whitespace with a single space
    text = re.sub(r'\s', ' ', text)
    # strip off spaces on either end
    text = text.strip()
    return text

class ScholarProcessor(ReaderPostProcessorBase):
    def __init__(self, x_fields=['Claim', 'Explaination'], y_field='selected_label', **kwargs):
        super().__init__(**kwargs)
        self.x_fields = x_fields
        self.y_field = y_field
        self.initProcessor()

    def initProcessor(self):
        bert_tokenizer_path = os.path.join(self.config['BERT'].get('bert_path'), 'tokenizer')
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
            current_x = self.x_pipeline(current_rawx, add_special_tokens=self.add_spec_tokens)

        ## NLTK tokenise and remove stopwords for topic modelling
        #current_x_nltk_tokened = self.nltkTokenizer(current_rawx)
        #current_x_nltk_tokened = self._remove_stop_words(current_x_nltk_tokened)
        current_x_nltk_tokened = tokenize(current_rawx)
        if self.dictProcess:
            current_x_nltk_tokened = self.dictProcess.doc2countHot(current_x_nltk_tokened)

        x=[current_x, current_x_nltk_tokened]

        y = sample[self.y_field]
        if self.label2id:
            y = self.label2ids(y)

        if self.remove_single_list:
            x = self._removeSingleList(x)
            y = self._removeSingleList(y)
        return x, y
