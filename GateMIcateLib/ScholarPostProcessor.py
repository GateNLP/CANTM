import nltk
from nltk.corpus import stopwords
import os
import re
from .WVPostProcessor import WVPostProcessor 
from transformers import BertTokenizer
import glob
import string
from collections import Counter
from nltk.tokenize import sent_tokenize



punct_chars = list(set(string.punctuation) - set("'"))
punct_chars.sort()
punctuation = ''.join(punct_chars)
replace = re.compile('[%s]' % re.escape(punctuation))

punct_chars_more = list(set(string.punctuation) - set(["'",","]))
punct_chars_more.sort()
punctuation_more = ''.join(punct_chars_more)
replace_more = re.compile('[%s]' % re.escape(punctuation_more))


alpha = re.compile('^[a-zA-Z_]+$')
alpha_or_num = re.compile('^[a-zA-Z_]+|[0-9_]+$')
alphanum = re.compile('^[a-zA-Z0-9_]+$')


def tokenize(text, strip_html=False, lower=True, keep_emails=False, keep_at_mentions=False, keep_numbers=False, keep_alphanum=False, min_length=3, stopwords=None, vocab=None, keep_pun=False):
    text = clean_text(text, strip_html, lower, keep_emails, keep_at_mentions, keep_pun=keep_pun)
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


def clean_text(text, strip_html=False, lower=True, keep_emails=False, keep_at_mentions=False, keep_pun=False):
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
    if not keep_pun:
        # remove periods
        text = re.sub(r'\.', '', text)
        # replace all other punctuation (except single quotes) with spaces
        text = replace.sub(' ', text)
    else:
        # remove periods
        text = re.sub(r'\.', '', text)
        # replace all other punctuation (except single quotes) with spaces
        text = replace_more.sub(' ', text)


    # remove single quotes
    text = re.sub(r'\'', '', text)
    # replace all whitespace with a single space
    text = re.sub(r'\s', ' ', text)
    # strip off spaces on either end
    text = text.strip()
    return text




class ScholarPostProcessor(WVPostProcessor):
    def __init__(self, stopwords_source=['snowball'], min_token_length=3, **kwargs):
        super().__init__(**kwargs)
        script_path = os.path.abspath(__file__)
        parent = os.path.dirname(script_path)
        self.stopwords_source = stopwords_source
        self.min_token_length = min_token_length
        stop_list_dir = os.path.join(parent, 'stopwords')
        self._get_stop_words(stop_list_dir)
        print(self.stop_words)

    def _get_stop_words(self, stop_list_dir):
        self.stop_words = set()

        snowball_stopwords_list_file = os.path.join(stop_list_dir, 'snowball_stopwords.txt')
        mallet_stopwords_list_file = os.path.join(stop_list_dir, 'mallet_stopwords.txt')
        scholar_stopwords_list_file = os.path.join(stop_list_dir, 'custom_stopwords.txt')

        if 'snowball' in self.stopwords_source:
            with open(snowball_stopwords_list_file, 'r') as fin:
                for line in fin:
                    stop_word = line.strip()
                    self.stop_words.add(stop_word)



    def clean_source(self, source_text):
        #split_lines = source_text.split('\n | _')
        split_lines = re.split('\n|_|=|\*|\||\/', source_text)
        added_sent = []
        for splited_line in split_lines:
            all_sents_split = sent_tokenize(splited_line)
            for each_sent in all_sents_split:
                keep = True
                line_tok, _= tokenize(splited_line, stopwords=self.stop_words)
                if len(line_tok) < 3:
                    keep=False
                if keep:
                    added_sent.append(splited_line)
        return ' '.join(added_sent)



    def postProcess(self, sample):
        split_x = []
        for x_field in self.x_fields:
            current_rawx = self._get_sample(sample, x_field)
            split_x.append(current_rawx)
        current_rawx = ' '.join(split_x)
        current_rawx_tokened, _ = tokenize(current_rawx, keep_numbers=True, keep_alphanum=True, min_length=1, keep_pun=True)
        current_rawx = ' '.join(current_rawx_tokened)
        #print(current_rawx)

        ## Bert toknise for hidden layers. add_special_tokens not added, additional attention will be applied on token level (CLS not used)
        if self.embd_ready:
            current_x = sample['embd']
        else:
            #cleaned_raw_x = self.clean_source(current_rawx)
            #print(cleaned_raw_x)
            cleaned_raw_x = current_rawx
            if len(cleaned_raw_x) > 10:
                current_x = self.x_pipeline(cleaned_raw_x, add_special_tokens=self.add_spec_tokens)
            else:
                current_x = self.x_pipeline(current_rawx, add_special_tokens=self.add_spec_tokens)

        ## NLTK tokenise and remove stopwords for topic modelling
        #current_x_nltk_tokened = self.nltkTokenizer(current_rawx)
        #current_x_nltk_tokened = self._remove_stop_words(current_x_nltk_tokened)
        current_x_nltk_tokened,_ = tokenize(current_rawx, stopwords=self.stop_words)
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

    def _remove_stop_words(self, tokened):
        remain_list = []
        for token in tokened:
            contain_symbol, contain_number, contain_char, all_asii = self._check_string(token)
            keep = True
            if token in self.stop_words:
                keep = False
            elif len(token) < self.min_token_length:
                keep = False
            elif token.isdigit():
                keep = False
            elif not contain_char:
                keep = False
            elif not all_asii:
                keep = False
            elif contain_number:
                keep = False
                
            if keep == True:
                remain_list.append(token)
            else:
                pass
                #print(token)
        return remain_list

