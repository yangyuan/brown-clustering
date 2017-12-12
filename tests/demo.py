import unittest
from brownclustering import Corpus
from brownclustering import BrownClustering
from brownclustering.validation import BrownClusteringValidation


class MyTestCase(unittest.TestCase):
    def test_something(self):

        data = [['i', 'have', 'a', 'dream', '.'],
                ['this', 'is', 'going', 'to', 'be', 'fun', '!'],
                ['i', 'have', 'a', 'dream', 'too', '.'],
                ['why', 'you', 'also', 'have', 'a', 'dream', '?']]
        corpus = Corpus(data, 1)

        self.assertEqual(corpus.n, sum(corpus.bigrams.values()))
        self.assertEqual(sum(corpus.unigrams.values()), sum(corpus.bigrams.values()) + len(data))

    def test_eval_something(self):

        data = [['i', 'have', 'a', 'dream', '.'],
                ['this', 'is', 'going', 'to', 'be', 'fun', '!'],
                ['i', 'have', 'a', 'dream', 'too', '.'],
                ['why', 'you', 'also', 'have', 'a', 'dream', '?']]
        corpus = Corpus(data, 0.001)
        print(corpus.bigrams)
        clustering = BrownClusteringValidation(corpus, 6)
        clustering.validate()

        self.assertAlmostEqual(corpus.n, sum(corpus.bigrams.values()), )
        self.assertAlmostEqual(sum(corpus.unigrams.values()), sum(corpus.bigrams.values()) + len(data))

    def test_full_something(self):

        data = [['i', 'have', 'a', 'dream', '.'],
                ['this', 'is', 'going', 'to', 'be', 'fun', '!'],
                ['i', 'have', 'a', 'dream', 'too', '.'],
                ['why', 'you', 'also', 'have', 'a', 'dream', '?']]
        corpus = Corpus(data, 0.001)
        clustering = BrownClustering(corpus, 6)
        clustering.train()
        clustering.get_similar('i')


if __name__ == '__main__':
    unittest.main()
