import math
import numpy as np


class ClusteringHelper:

    def __init__(self, corpus):
        self.n = corpus.n
        self.unigrams = corpus.unigrams
        self.bigrams = corpus.bigrams

    def count_bigrams(self, cluster1, cluster2):
        _count = 0
        for w1 in cluster1:
            for w2 in cluster2:
                _count += self.bigrams.get((w1, w2), 0)
        return _count

    def append_cluster(self, words):
        raise NotImplementedError()

    def merge_clusters(self, i, j):
        raise NotImplementedError()

    def get_clusters(self):
        raise NotImplementedError()

    def get_cluster(self, i):
        raise NotImplementedError()

    def compute_benefit(self):
        raise NotImplementedError()


class EnhancedClusteringHelper(ClusteringHelper):
    def __init__(self, corpus):
        super().__init__(corpus)
        self.m = 0
        self.clusters = []
        self.p1 = np.zeros(0, dtype=float)
        self.p2 = np.zeros((0, 0), dtype=float)
        self.q2 = np.zeros((0, 0), dtype=float)
        self.l2 = np.zeros((0, 0), dtype=float)

    def append_cluster(self, words):
        """
        O(m (+ n))
        """

        self.p1 = np.insert(self.p1, self.m, 0, axis=0)
        _sum = 0
        for w in words:
            _sum += self.unigrams[w]
        self.p1[self.m] = _sum / self.n

        self.p2 = np.insert(self.p2, self.m, 0, axis=1)
        self.p2 = np.insert(self.p2, self.m, 0, axis=0)
        for i in range(self.m):
            self.p2[self.m, i] = self.count_bigrams(words, self.clusters[i]) / self.n
            self.p2[i, self.m] = self.count_bigrams(self.clusters[i], words) / self.n
        self.p2[self.m, self.m] = self.count_bigrams(words, words) / self.n

        self.q2 = np.insert(self.q2, self.m, 0, axis=1)
        self.q2 = np.insert(self.q2, self.m, 0, axis=0)
        for i in range(self.m):
            self.q2[self.m, i] = self._q(self.m, i)
            self.q2[i, self.m] = self._q(i, self.m)
        self.q2[self.m, self.m] = self._q(self.m, self.m)

        self.l2 = np.insert(self.l2, self.m, 0, axis=1)
        self.l2 = np.insert(self.l2, self.m, 0, axis=0)

        self.m = self.m + 1
        self.clusters.append(words.copy())

        self._update_deltas()

    def _update_deltas(self):

        for i in range(self.m - 1):
            self.l2[i, self.m - 1] = self._delta(i, self.m - 1)
            for j in range(i + 1, self.m - 1):
                self.l2[i, j] -= self.q2[i, self.m - 1]
                self.l2[i, j] -= self.q2[j, self.m - 1]
                self.l2[i, j] -= self.q2[self.m - 1, i]
                self.l2[i, j] -= self.q2[self.m - 1, j]
                self.l2[i, j] += self._q_l(i, j, self.m - 1)
                self.l2[i, j] += self._q_r(self.m - 1, i, j)

    def get_clusters(self):
        return self.clusters.copy()

    def get_cluster(self, i):
        return self.clusters[i].copy()

    def merge_clusters(self, i, j):

        for _i in range(self.m):
            for _j in range(_i+1, self.m):
                _tmp = 0
                _tmp += self._q_l(_i, _j, i)
                _tmp += self._q_l(_i, _j, j)
                _tmp += self._q_r(i, _i, _j)
                _tmp += self._q_r(j, _i, _j)

                _tmp -= self.q2[i, _i]
                _tmp -= self.q2[_i, i]
                _tmp -= self.q2[i, _j]
                _tmp -= self.q2[_j, i]

                _tmp -= self.q2[j, _i]
                _tmp -= self.q2[_i, j]
                _tmp -= self.q2[j, _j]
                _tmp -= self.q2[_j, j]

                self.l2[_i, _j] -= _tmp

        self.clusters[i].extend(self.clusters[j])
        del self.clusters[j]
        self.m = self.m - 1

        self.p1[i] = self.p1[i] + self.p1[j]
        self.p1 = np.delete(self.p1, j, axis=0)

        self.p2[i, :] = self.p2[i, :] + self.p2[j, :]
        self.p2[:, i] = self.p2[:, i] + self.p2[:, j]
        self.p2 = np.delete(self.p2, j, axis=0)
        self.p2 = np.delete(self.p2, j, axis=1)

        self.q2 = np.delete(self.q2, j, axis=0)
        self.q2 = np.delete(self.q2, j, axis=1)
        for _x in range(self.m):
            self.q2[i, _x] = self._q(i, _x)
            self.q2[_x, i] = self._q(_x, i)
        self.q2[i, i] = self._q(i, i)

        self.l2 = np.delete(self.l2, j, axis=0)
        self.l2 = np.delete(self.l2, j, axis=1)

        for _i in range(self.m):
            for _j in range(_i+1, self.m):
                _tmp = 0
                _tmp += self._q_l(_i, _j, i)
                _tmp += self._q_r(i, _i, _j)

                _tmp -= self.q2[i, _i]
                _tmp -= self.q2[_i, i]
                _tmp -= self.q2[i, _j]
                _tmp -= self.q2[_j, i]

                self.l2[_i, _j] += _tmp

        for x in range(i):
            # print("%d %d " % (x, i))
            self.l2[x, i] = self._delta(x, i)
        for x in range(i+1, self.m):
            # print("%d %d " % (i, x))
            self.l2[i, x] = self._delta(i, x)

    def compute_benefit(self):
        return self.l2.copy()

    def _q_l(self, _i, _j, _x):
        """
        O(1)
        """
        pcx = (self.p2[_i, _x] + self.p2[_j, _x])
        pc = (self.p1[_i] + self.p1[_j])
        px = self.p1[_x]

        return pcx * math.log(pcx / (pc * px))

    def _q_r(self, _x, _i, _j):
        """
        O(1)
        """
        pxc = (self.p2[_x, _i] + self.p2[_x, _j])
        pc = (self.p1[_i] + self.p1[_j])
        px = self.p1[_x]
        return pxc * math.log(pxc / (pc * px))

    def _q_x(self, _i, _j):
        """
        O(1)
        """
        pxc = (self.p2[_j, _i] + self.p2[_i, _j] + self.p2[_i, _i] + self.p2[_j, _j])
        pc = (self.p1[_i] + self.p1[_j])
        px = (self.p1[_i] + self.p1[_j])
        return pxc * math.log(pxc / (pc * px))

    def _q(self, _i, _x):
        """
        O(1)
        """
        pcx = self.p2[_i, _x]
        pc = self.p1[_i]
        px = self.p1[_x]

        return pcx * math.log(pcx / (pc * px))

    def _delta(self, i, j):
        count_i_new = self.p1[i] + self.p1[j]
        count_2_new_s = self.p2[i, :] + self.p2[j, :]
        count_2_new_e = self.p2[:, i] + self.p2[:, j]

        # O(1)
        def _weight_new_1(_x):
            pij = count_2_new_s[_x]
            pji = count_2_new_e[_x]
            pi = count_i_new
            pj = self.p1[_x]
            return pij * math.log(pij / (pi * pj)) + pji * math.log(pji / (pi * pj))

        # O(1)
        def _weight_new_2():
            pij = (self.p2[i, i] + self.p2[i, j] + self.p2[j, i] + self.p2[j, j])
            pji = pij
            pi = count_i_new
            pj = count_i_new
            return pij * math.log(pij / (pi * pj)) + pji * math.log(pji / (pi * pj))

        # O(m)
        loss = 0
        for x in range(self.m):
            loss -= self.q2[i, x]
            loss -= self.q2[x, i]
            loss -= self.q2[j, x]
            loss -= self.q2[x, j]
            if x == i or x == j:
                continue
            loss += _weight_new_1(x)
        loss += _weight_new_2() / 2
        loss += self.q2[i, j]
        loss += self.q2[j, i]
        loss += self.q2[i, i]
        loss += self.q2[j, j]

        return loss


class RawClusteringHelper(ClusteringHelper):
    def __init__(self, corpus):
        super().__init__(corpus)
        self.clusters = []

    def append_cluster(self, words):
        self.clusters.append(words.copy())

    def merge_clusters(self, i, j):
        self.clusters[i].extend(self.clusters[j])
        del self.clusters[j]

    def get_clusters(self):
        return self.clusters.copy()

    def get_cluster(self, i):
        return self.clusters[i].copy()

    def count(self, c1, c2=None):
        ret = 0
        if c2 is None:
            for w in c1:
                ret += self.unigrams[w]
        else:
            for w1 in c1:
                for w2 in c2:
                    ret += self.bigrams[w1, w2]
        return ret

    def compute_average_mutual_information(self, clusters):
        k = len(clusters)
        counts = dict()
        for i in range(k):
            counts[i] = self.count(clusters[i])
        for i in range(k):
            for j in range(k):
                counts[(i, j)] = self.count(clusters[i], clusters[j])

        def _prob(_i, _j=None):
            if _j is None:
                return counts[_i]/self.n
            return counts[(_i, _j)]/self.n

        ret = 0
        for i in range(k):
            for j in range(k):
                ret += _prob(i, j) * math.log(_prob(i, j) / (_prob(i) * _prob(j)))
        return ret

    def compute_benefit(self):
        tmp = self.compute_average_mutual_information(self.clusters)


        k = len(self.clusters)

        _benefit = np.zeros((k, k))
        for i in range(k):
            for j in range(i + 1, k):
                clusters_candidate = [x.copy() for x in self.clusters]
                clusters_candidate[i].extend(clusters_candidate[j])
                del clusters_candidate[j]
                _benefit[i, j] = self.compute_average_mutual_information(clusters_candidate) - tmp
        return _benefit


class ModerateClusteringHelper(ClusteringHelper):
    def __init__(self, corpus):
        super().__init__(corpus)
        self.m = 0
        self.clusters = []
        self.counts_1 = np.zeros(0, dtype=float)
        self.counts_2 = np.zeros((0, 0), dtype=float)

    def append_cluster(self, words):
        """
        O(m (+ n))
        """

        self.counts_1 = np.insert(self.counts_1, self.m, 0, axis=0)
        _sum = 0
        for w in words:
            _sum += self.unigrams[w]
        self.counts_1[self.m] = _sum

        self.counts_2 = np.insert(self.counts_2, self.m, 0, axis=1)
        self.counts_2 = np.insert(self.counts_2, self.m, 0, axis=0)
        for i in range(self.m):
            self.counts_2[self.m, i] = self.count_bigrams(words, self.clusters[i])
            self.counts_2[i, self.m] = self.count_bigrams(self.clusters[i], words)
        self.counts_2[self.m, self.m] = self.count_bigrams(words, words)

        self.m = self.m + 1
        self.clusters.append(words.copy())

    def get_clusters(self):
        return self.clusters.copy()

    def get_cluster(self, i):
        return self.clusters[i].copy()

    def merge_clusters(self, i, j):
        self.counts_1[i] = self.counts_1[i] + self.counts_1[j]
        self.counts_1 = np.delete(self.counts_1, j, axis=0)

        self.counts_2[i, :] = self.counts_2[i, :] + self.counts_2[j, :]
        self.counts_2[:, i] = self.counts_2[:, i] + self.counts_2[:, j]
        self.counts_2 = np.delete(self.counts_2, j, axis=0)
        self.counts_2 = np.delete(self.counts_2, j, axis=1)

        self.clusters[i].extend(self.clusters[j])
        del self.clusters[j]
        self.m = self.m - 1

    def compute_benefit(self):
        deltas = np.zeros((self.m, self.m), dtype=float)
        for i in range(self.m):
            for j in range(i + 1, self.m):
                deltas[i, j] = self._delta(i, j)
        return deltas

    def _q_l(self, _i, _j, _x):
        """
        O(1)
        """
        pcx = (self.counts_2[_i, _x] + self.counts_2[_j, _x]) / self.n
        pc = (self.counts_1[_i] + self.counts_1[_j]) / self.n
        px = self.counts_1[_x] / self.n

        return pcx * math.log(pcx / (pc * px))

    def _q_r(self, _x, _i, _j):
        """
        O(1)
        """
        pxc = (self.counts_2[_x, _i] + self.counts_2[_x, _j]) / self.n
        pc = (self.counts_1[_i] + self.counts_1[_j]) / self.n
        px = self.counts_1[_x] / self.n
        return pxc * math.log(pxc / (pc * px))

    def _q_x(self, _i, _j):
        """
        O(1)
        """
        pxc = (self.counts_2[_j, _i] + self.counts_2[_i, _j] + self.counts_2[_i, _i] + self.counts_2[_j, _j]) / self.n
        pc = (self.counts_1[_i] + self.counts_1[_j]) / self.n
        px = (self.counts_1[_i] + self.counts_1[_j]) / self.n
        return pxc * math.log(pxc / (pc * px))

    def _q(self, _i, _x):
        """
        O(1)
        """
        pcx = self.counts_2[_i, _x] / self.n
        pc = self.counts_1[_i] / self.n
        px = self.counts_1[_x] / self.n

        return pcx * math.log(pcx / (pc * px))

    def _delta(self, i, j):
        """
        O(m)
        """

        # O(m)

        # O(m)
        count_i_new = self.counts_1[i] + self.counts_1[j]
        count_2_new_s = self.counts_2[i, :] + self.counts_2[j, :]
        count_2_new_e = self.counts_2[:, i] + self.counts_2[:, j]

        # O(1)
        def _weight_new_1(_x):
            pij = count_2_new_s[_x] / self.n
            pji = count_2_new_e[_x] / self.n
            pi = count_i_new / self.n
            pj = self.counts_1[_x] / self.n
            return pij * math.log(pij / (pi * pj)) + pji * math.log(pji / (pi * pj))

        # O(1)
        def _weight_new_2():
            pij = (self.counts_2[i, i] + self.counts_2[i, j] + self.counts_2[j, i] + self.counts_2[j, j]) / self.n
            pji = pij
            pi = count_i_new / self.n
            pj = count_i_new / self.n
            return pij * math.log(pij / (pi * pj)) + pji * math.log(pji / (pi * pj))

        # O(m)
        loss = 0
        for x in range(self.m):
            loss -= self._q(i, x)
            loss -= self._q(x, i)
            loss -= self._q(j, x)
            loss -= self._q(x, j)
            if x == i or x == j:
                continue
            loss += _weight_new_1(x)
        loss += _weight_new_2() / 2
        loss += self._q(i, j)
        loss += self._q(j, i)
        loss += self._q(i, i)
        loss += self._q(j, j)

        return loss
