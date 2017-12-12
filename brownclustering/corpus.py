from nltk.util import ngrams


class Corpus:
    def __init__(self, corpus, alpha=1, start_symbol='<s>', end_symbol='</s>'):
        self.n = 0
        self.vocabulary = dict()
        self.unigrams = dict()
        self.bigrams = dict()

        for sentence in corpus:
            for word in sentence:
                self.vocabulary[word] = self.vocabulary.get(word, 0) + 1
                self.unigrams[word] = self.unigrams.get(word, 0) + 1

            self.unigrams[start_symbol] = self.unigrams.get(start_symbol, 0) + 1
            self.unigrams[end_symbol] = self.unigrams.get(end_symbol, 0) + 1

            grams = ngrams([start_symbol] + sentence + [end_symbol], 2)
            for gram in grams:
                self.n += 1
                if gram in self.bigrams:
                    self.bigrams[gram] += 1
                else:
                    self.bigrams[gram] = 1

        # Laplace smoothing
        _vocabulary = list(self.vocabulary.keys()) + [start_symbol, end_symbol]
        for w in _vocabulary:
            for w2 in _vocabulary:
                self.n += alpha
                self.bigrams[w, w2] = self.bigrams.get((w, w2), 0) + alpha
                self.unigrams[w2] += alpha
