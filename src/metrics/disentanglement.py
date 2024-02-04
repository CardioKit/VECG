from sklearn.preprocessing import KBinsDiscretizer
from sklearn.metrics import mutual_info_score
import numpy as np
from sklearn import svm


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
    def compute_score_matrix(mus, ys, mus_test, ys_test, continuous_factors):
        """Compute score matrix as described in Section 3."""
        num_latents = mus.shape[0]
        num_factors = ys.shape[0]
        score_matrix = np.zeros([num_latents, num_factors])
        for i in range(num_latents):
            for j in range(num_factors):
                mu_i = mus[i, :]
                y_j = ys[j, :]
                if continuous_factors:
                    # Attribute is considered continuous.
                    cov_mu_i_y_j = np.cov(mu_i, y_j, ddof=1)
                    cov_mu_y = cov_mu_i_y_j[0, 1] ** 2
                    var_mu = cov_mu_i_y_j[0, 0]
                    var_y = cov_mu_i_y_j[1, 1]
                    if var_mu > 1e-12:
                        score_matrix[i, j] = cov_mu_y * 1. / (var_mu * var_y)
                    else:
                        score_matrix[i, j] = 0.
                else:
                    # Attribute is considered discrete.
                    mu_i_test = mus_test[i, :]
                    y_j_test = ys_test[j, :]
                    classifier = svm.LinearSVC(C=0.01, class_weight="balanced")
                    classifier.fit(mu_i[:, np.newaxis], y_j)
                    pred = classifier.predict(mu_i_test[:, np.newaxis])
                    score_matrix[i, j] = np.mean(pred == y_j_test)
        return score_matrix

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

    @staticmethod
    def compute_avg_diff_top_two(matrix):
        sorted_matrix = np.sort(matrix, axis=0)
        return np.mean(sorted_matrix[-1, :] - sorted_matrix[-2, :])

    @staticmethod
    def _compute_sap(mus, ys, mus_test, ys_test, continuous_factors):
        """Computes score based on both training and testing codes and factors."""
        score_matrix = Disentanglement.compute_score_matrix(mus, ys, mus_test,
                                            ys_test, continuous_factors)
        # Score matrix should have shape [num_latents, num_factors].
        assert score_matrix.shape[0] == mus.shape[0]
        assert score_matrix.shape[1] == ys.shape[0]
        scores_dict = {}
        scores_dict["SAP_score"] = Disentanglement.compute_avg_diff_top_two(score_matrix)
        return scores_dict
