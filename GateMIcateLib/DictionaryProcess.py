class DictionaryProcess:
    def __init__(self, common_dictionary):
        self.common_dictionary = common_dictionary
        self.num_vocab = len(self.common_dictionary)

    def doc2bow(self, input_doc):
        gensim_bow_doc = self.common_dictionary.doc2bow(input_doc)
        return gensim_bow_doc

    def doc2countHot(self, input_doc):
        gensim_bow_doc = self.doc2bow(input_doc)
        doc_vec = [0] * self.num_vocab
        for item in gensim_bow_doc:
            vocab_idx = item[0]
            vovab_counts = item[1]
            doc_vec[vocab_idx] = vovab_counts
        return doc_vec

    def get(self, wordidx):
        return self.common_dictionary[wordidx]

    def __len__(self):
        return self.num_vocab
