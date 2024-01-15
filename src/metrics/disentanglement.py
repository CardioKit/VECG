from sklearn.preprocessing import KBinsDiscretizer
from sklearn.metrics import mutual_info_score
import numpy as np


class Disentanglement:

    @staticmethod
    def _discrete_mutual_info(mus, ys):
        """Compute discrete mutual information."""
        num_codes = mus.shape[1]
        num_factors = ys.shape[1]
        m = np.zeros([num_codes, num_factors])
        for i in range(num_codes):
            for j in range(num_factors):
                m[i, j] = mutual_info_score(ys[:, j], mus[:, i])
        return m

    @staticmethod
    def _discrete_entropy(ys):
        """Compute discrete mutual information."""
        num_factors = ys.shape[1]
        h = np.zeros(num_factors)
        for j in range(num_factors):
            h[j] = mutual_info_score(ys[:, j], ys[:, j])
        return h

    @staticmethod
    def _make_discretizer(target, num_bins):
        dis = KBinsDiscretizer(num_bins, encode='ordinal').fit(target)
        return dis.transform(target)

    @staticmethod
    def compute_mig(mus_train, ys_train, num_bins=10):
        """Computes score based on both training and testing codes and factors."""
        score_dict = {}
        discrete_mus = Disentanglement._make_discretizer(mus_train, num_bins).astype(int)
        discrete_ys = Disentanglement._make_discretizer(ys_train, num_bins).astype(int)
        m = Disentanglement._discrete_mutual_info(discrete_mus, discrete_ys)
        # m is [num_latents, num_factors]
        assert m.shape[0] == mus_train.shape[1]
        assert m.shape[1] == ys_train.shape[1]
        entropy = Disentanglement._discrete_entropy(discrete_ys)
        sorted_m = np.sort(m, axis=0)[::-1]
        score_dict["discrete_mig"] = np.mean(np.divide(sorted_m[0, :] - sorted_m[1, :], entropy[:]))
        return score_dict
