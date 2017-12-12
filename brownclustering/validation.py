from brownclustering.helpers import *
from brownclustering.core import *


class BrownClusteringValidation(BrownClustering):
    def __init__(self, corpus, m):
        super().__init__(corpus, m)

        self.helpers = []
        self.helpers.append(RawClusteringHelper(corpus))
        self.helpers.append(ModerateClusteringHelper(corpus))
        self.helpers.append(EnhancedClusteringHelper(corpus))

    @staticmethod
    def arg_max(_benefit, _helper):
        max_benefit = float('-inf')
        best_merge = None
        for i in range(_benefit.shape[0]):
            for j in range(i + 1, _benefit.shape[1]):
                if max_benefit < _benefit[i, j]:
                    max_benefit = _benefit[i, j]
                    best_merge = (i, j)
        return best_merge

    def validate(self):
        words = self.ranks(self.vocabulary)
        tops = words[0:self.m]

        count = 0
        _words = []
        for w in tops:
            _words.append(w[0])
            if len(_words) > 2:
                for helper in self.helpers:
                    helper.append_cluster(_words)
                _words = []
                count += 1

        if len(_words) > 0:
            for helper in self.helpers:
                helper.append_cluster(_words)
            count += 1

        itr = 0
        for w in words[self.m:]:
            itr += 1
            print(str(itr) + "\t" + str(datetime.datetime.now()))

            last_benefit = None
            for helper in self.helpers:
                helper.append_cluster([w[0]])
                _benefit = helper.compute_benefit()
                if last_benefit is not None:
                    print(_benefit - last_benefit)
                last_benefit = _benefit
                best_merge = self.arg_max(_benefit, helper)

            print(best_merge)
            for helper in self.helpers:
                helper.merge_clusters(best_merge[0], best_merge[1])

        for _ in range(count - 1):
            itr += 1
            print(str(itr) + "\t" + str(datetime.datetime.now()))
            last_benefit = None
            for helper in self.helpers:
                _benefit = helper.compute_benefit()
                if last_benefit is not None:
                    print(_benefit - last_benefit)
                best_merge = self.arg_max(_benefit, helper)

            print(best_merge)
            for helper in self.helpers:
                helper.merge_clusters(best_merge[0], best_merge[1])