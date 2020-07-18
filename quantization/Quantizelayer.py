from __future__ import division
from __future__ import print_function
import numpy as np
import copy
from scipy import stats


class QuantizeLayer:
    def __init__(self, name="None", num_bin=2001):
        self.name = name
        self.min = 0.0
        self.max = 0.0
        self.edge = 0.0
        self.num_bins = num_bin
        self.distribution_interval = 0.0
        self.data_distribution = []

    @staticmethod
    def get_max_min_edge(blob_data):
        max_val = np.max(blob_data)
        min_val = np.min(blob_data)
        data_edge = max(abs(max_val), abs(min_val))
        return max_val, min_val, data_edge

    def initial_histograms(self, blob_data):
        max_val, min_val, data_edge = self.get_max_min_edge(blob_data)
        hist, hist_edges = np.histogram(blob_data, bins=self.num_bins, range=(-data_edge, data_edge))
        self.distribution_interval = 2 * data_edge / len(hist)
        self.data_distribution = hist
        self.edge = data_edge
        self.min = min_val
        self.max = max_val

    def combine_histograms(self, blob_data):
        """
        :param blob_data:
        :return:
        """
        # hist is the num of each bin,  the edge of each bin is [)
        max_val, min_val, data_edge = self.get_max_min_edge(blob_data)
        if data_edge <= self.edge:
            hist, _ = np.histogram(blob_data, bins=len(self.data_distribution), range=(-self.edge, self.edge))
            self.data_distribution += hist
        else:
            old_num_bins = len(self.data_distribution)
            old_step = 2 * self.edge / old_num_bins
            half_increased_bins = int((data_edge - self.edge) // old_step + 1)
            new_num_bins = half_increased_bins * 2 + old_num_bins
            data_edge = half_increased_bins * old_step + self.edge
            hist, hist_edges = np.histogram(blob_data, bins=new_num_bins, range=(-data_edge, data_edge))
            hist[half_increased_bins:new_num_bins - half_increased_bins] += self.data_distribution
            self.data_distribution = hist
            self.edge = data_edge
        self.min = min(min_val, self.min)
        self.max = max(max_val, self.max)
        self.distribution_interval = 2 * self.edge / len(self.data_distribution)

    @staticmethod
    def smooth_distribution(p, eps=0.0001):

        is_zeros = (p == 0).astype(np.float32)
        is_nonzeros = (p != 0).astype(np.float32)
        n_zeros = is_zeros.sum()
        n_nonzeros = p.size - n_zeros
        if not n_nonzeros:
            raise ValueError('The discrete probability distribution is malformed. All entries are 0.')
        eps1 = eps * float(n_zeros) / float(n_nonzeros)
        assert eps1 < 1.0, 'n_zeros=%d, n_nonzeros=%d, eps1=%f' % (n_zeros, n_nonzeros, eps1)
        hist = p.astype(np.float32)
        hist += eps * is_zeros + (-eps1) * is_nonzeros
        assert (hist <= 0).sum() == 0
        return hist

    @property
    def threshold_distribution(self, target_bin=256):
        """
        :param quantized_dtype:
        :param target_bin:
        :return:
        """
        num_bins = len(self.data_distribution)
        distribution = self.data_distribution
        assert (num_bins % 2 == 1)

        # if min_val >= 0 and quantized_dtype in ['auto', 'uint8']:
        #     target_bin = 128

        threshold_sum = sum(distribution[target_bin:])
        kl_divergence = np.zeros(num_bins - target_bin)

        for threshold in range(target_bin, num_bins):
            sliced_nd_hist = copy.deepcopy(distribution[:threshold])

            # generate reference distribution p
            p = sliced_nd_hist.copy()
            p[threshold - 1] += threshold_sum
            threshold_sum = threshold_sum - distribution[threshold]

            # is_nonzeros[k] indicates whether hist[k] is nonzero
            p = np.array(p)
            nonzero_loc = (p != 0).astype(np.int64)
            #
            quantized_bins = np.zeros(target_bin, dtype=np.int64)
            # calculate how many bins should be merged to generate quantized distribution q
            num_merged_bins = len(sliced_nd_hist) // target_bin

            # merge hist into num_quantized_bins bins
            for j in range(target_bin):
                start = j * num_merged_bins
                stop = start + num_merged_bins
                quantized_bins[j] = sliced_nd_hist[start:stop].sum()
            quantized_bins[-1] += sliced_nd_hist[target_bin * num_merged_bins:].sum()

            # expand quantized_bins into p.size bins
            q = np.zeros(sliced_nd_hist.size, dtype=np.float64)
            for j in range(target_bin):
                start = j * num_merged_bins
                if j == target_bin - 1:
                    stop = -1
                else:
                    stop = start + num_merged_bins
                norm = nonzero_loc[start:stop].sum()
                if norm != 0:
                    q[start:stop] = quantized_bins[j] / norm

            q[p == 0] = 0.0001
            p = self.smooth_distribution(p)

            # calculate kl_divergence between q and p
            kl_divergence[threshold - target_bin] = stats.entropy(p, q)

        min_kl_divergence = np.argmin(kl_divergence)

        threshold_bin = min_kl_divergence + target_bin

        threshold_value = (threshold_bin + 0.5) * self.distribution_interval + (-self.edge)
        return threshold_value

    @staticmethod
    def max_slide_window(seq, m):
        num = len(seq)
        seq = seq.tolist()
        assert isinstance(seq, (list, tuple, set)) and isinstance(m, int), "seq array"
        assert len(seq) > m, "len(seq) must >m"
        max_seq = 0
        loc = 0
        for i in range(0, num):
            if (i + m) <= num:
                temp_seq = seq[i:i + m]
                temp_sum = sum(temp_seq)
                if max_seq <= temp_sum:
                    max_seq = temp_sum
                    loc = i
            else:
                return max_seq, loc

    @property
    def distribution_min_max(self, target_bin=256):
        num_bins = len(self.data_distribution)
        distribution = self.data_distribution
        assert (num_bins % 2 == 1)

        kl_divergence = np.zeros(num_bins - target_bin)
        kl_loc = np.zeros(num_bins - target_bin)

        for threshold in range(target_bin, num_bins):
            #print("num:", threshold)

            _, loc = self.max_slide_window(distribution, threshold)

            sliced_nd_hist = copy.deepcopy(distribution[loc:loc + threshold])

            # generate reference distribution p
            p = sliced_nd_hist.copy()
            right_sum = sum(distribution[loc + threshold:])
            left_sum = sum(distribution[:loc])
            p[threshold - 1] += right_sum
            p[0] += left_sum


            # is_nonzeros[k] indicates whether hist[k] is nonzero
            p = np.array(p)
            nonzero_loc = (p != 0).astype(np.int64)
            #
            quantized_bins = np.zeros(target_bin, dtype=np.int64)
            # calculate how many bins should be merged to generate quantized distribution q
            num_merged_bins = len(sliced_nd_hist) // target_bin

            # merge hist into num_quantized_bins bins
            for j in range(target_bin):
                start = j * num_merged_bins
                stop = start + num_merged_bins
                quantized_bins[j] = sliced_nd_hist[start:stop].sum()
            quantized_bins[-1] += sliced_nd_hist[target_bin * num_merged_bins:].sum()

            # expand quantized_bins into p.size bins
            q = np.zeros(sliced_nd_hist.size, dtype=np.float64)
            for j in range(target_bin):
                start = j * num_merged_bins
                if j == target_bin - 1:
                    stop = -1
                else:
                    stop = start + num_merged_bins
                norm = nonzero_loc[start:stop].sum()
                if norm != 0:
                    q[start:stop] = quantized_bins[j] / norm

            q[p == 0] = 0.0001
            p = self.smooth_distribution(p)

            # calculate kl_divergence between q and p
            kl_divergence[threshold - target_bin] = stats.entropy(p, q)
            kl_loc[threshold - target_bin] = loc

        min_kl_divergence = np.argmin(kl_divergence)

        min = kl_loc[min_kl_divergence]
        max = min + target_bin + min_kl_divergence
        min = (min + 0.5) * self.distribution_interval + (-self.edge)
        max = (max + 0.5) * self.distribution_interval + (-self.edge)
        return min, max

    @property
    def distribution_test(self, target_bin=256):
        num_bins = len(self.data_distribution)
        distribution = self.data_distribution
        assert (num_bins % 2 == 1)

        kl_divergence = np.zeros(num_bins - target_bin)
        kl_loc = np.zeros(num_bins - target_bin)

        for threshold in range(target_bin, num_bins):
            #print("num:", threshold)

            _, loc = self.max_slide_window(distribution, threshold)

            sliced_nd_hist = copy.deepcopy(distribution[loc:loc + threshold])

            # generate reference distribution p
            p = sliced_nd_hist.copy()
            right_sum = sum(distribution[loc + threshold:])
            left_sum = sum(distribution[:loc])
            p[threshold - 1] += right_sum
            p[0] += left_sum


            # is_nonzeros[k] indicates whether hist[k] is nonzero
            p = np.array(p)
            nonzero_loc = (p != 0).astype(np.int64)
            #
            quantized_bins = np.zeros(target_bin, dtype=np.int64)
            # calculate how many bins should be merged to generate quantized distribution q
            num_merged_bins = len(sliced_nd_hist) // target_bin

            # merge hist into num_quantized_bins bins
            for j in range(target_bin):
                start = j * num_merged_bins
                stop = start + num_merged_bins
                quantized_bins[j] = sliced_nd_hist[start:stop].sum()
            quantized_bins[-1] += sliced_nd_hist[target_bin * num_merged_bins:].sum()

            # expand quantized_bins into p.size bins
            q = np.zeros(sliced_nd_hist.size, dtype=np.float64)
            for j in range(target_bin):
                start = j * num_merged_bins
                if j == target_bin - 1:
                    stop = -1
                else:
                    stop = start + num_merged_bins
                norm = nonzero_loc[start:stop].sum()
                if norm != 0:
                    q[start:stop] = quantized_bins[j] / norm

            q[p == 0] = 0.0001
            p = self.smooth_distribution(p)

            # calculate kl_divergence between q and p
            kl_divergence[threshold - target_bin] = stats.wasserstein_distance(p, q)
            kl_loc[threshold - target_bin] = loc

        min_kl_divergence = np.argmin(kl_divergence)

        min = kl_loc[min_kl_divergence]
        max = min + target_bin + min_kl_divergence
        min = (min + 0.5) * self.distribution_interval + (-self.edge)
        max = (max + 0.5) * self.distribution_interval + (-self.edge)
        return min, max
data = np.random.randn(10000,)
print(data)
layer = QuantizeLayer(name="con_1")
layer.initial_histograms(data)
print("min:", layer.min)
print("max:", layer.max)
print("edge:", layer.edge)
print("distribution_interval:", layer.distribution_interval)
print("bins:", len(layer.data_distribution))
data = np.random.randn(10000,).astype()
layer.combine_histograms(data)
print("min:", layer.min)
print("max:", layer.max)
print("edge:", layer.edge)
print("distribution_interval:", layer.distribution_interval)
print("bins:", len(layer.data_distribution))

data = np.random.randn(10000,)
data[9999] = 20
layer.combine_histograms(data)
print("min:", layer.min)
print("max:", layer.max)
print("edge:", layer.edge)
print("distribution_interval:", layer.distribution_interval)
print("bins:", len(layer.data_distribution))

import matplotlib.pyplot as plt
plt.plot(layer.data_distribution)
plt.show()

print(layer.threshold_distribution)
print(layer.distribution_min_max)
#print(layer.distribution_test)