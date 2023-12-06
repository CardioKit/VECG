import numpy as np


# INFORMATION-BASED DISENTANGLEMENT METRICS #
class DisentanglementMetricsInfo:
    """
    Class that implements information-based disentanglement metrics
    """

    def __init__(self, sim_measure, dis_metric, mi_scores):
        """

        Args:
            sim_measure (SimilarityMeasure): an object containing DataPreprocessor data, a similarity measure
                and num_bins
            dis_metric (function): a disentanglement metric
            mi_scores (np.ndarray): array of shape (D,F) where each value corresponds to the mutual information between
                the generative factor and the latent dimension
        """
        self.sim_measure = sim_measure
        self.dis_metric = dis_metric
        self.mi_scores = mi_scores

    def compute_score(self):
        return self.dis_metric(self)

    def get_meta_data_entropies(self):
        return get_entropies(self.sim_measure.data.meta_data.values,
                             self.sim_measure.sim_function,
                             discrete_features=self.sim_measure.data.discrete_features,
                             num_bins=self.sim_measure.num_bins)

    def get_z_entropies(self):
        return get_entropies(self.sim_measure.data.z,
                             self.sim_measure.sim_function,
                             num_bins=self.sim_measure.num_bins)


def get_entropies(matrix, sim_function, discrete_features=False, num_bins=16):
    """
    compute entropy for features of meta_data
    exploit that the mutual of a random variable with itself is its entropy: I(X;X)=H(X)
    (Cover, Thomas M., and Joy A. Thomas. "Entropy, relative entropy and mutual information."
    Elements of information theory 2.1 (1991): 12-13.)

    Args:
        matrix (): array of shape (N,F) including continuous values for each factor and each data sample
        sim_function (function): a similarity measure
        discrete_features (): list of bool indicating which factors are discrete (True) or continuous (False)
        num_bins (int): the number of bins in which the continuous values are discretized

    Returns:
        np.ndarray: array of shape (F,) where each value corresponds to the factor's entropy

    """
    mi_scores = sim_function(matrix, matrix, discrete_features=discrete_features, num_bins=num_bins)
    entropies = np.diag(mi_scores)
    return entropies


def mutual_information_gap(dis_metric_object):
    """
    compute the Mutual Information Gap from a given matrix of Mutual Information and Entropies
    MIG = 1 / H(v_i) * I(v_i, z*) - I(v_i,z°)

    Args:
        dis_metric_object (DisentanglementMetricsInfo): a DisentanglementMetricsInfo object

    Returns:
        float: a single MIG value (good value: high)

    """
    mi_scores = dis_metric_object.mi_scores
    entropies = dis_metric_object.get_meta_data_entropies()

    num_factors = mi_scores.shape[1]

    gap = 0
    for factor in range(num_factors):
        mi_f_sorted = np.flip(np.sort(mi_scores[:, factor]))
        gap += (mi_f_sorted[0] - mi_f_sorted[1]) / entropies[factor]
    gap /= num_factors

    return gap


def rmig(dis_metric_object):
    """
    compute the Robust Mutual Information Gap (RMIG) from a given matrix of Mutual Information
    RMIG = I(v_i, z*) - I(v_i,z°)

     Args:
        dis_metric_object (DisentanglementMetricsInfo): a DisentanglementMetricsInfo object

    Returns:
        float: a single MIG value (good value: high)

    """
    mi_scores = dis_metric_object.mi_scores

    num_factors = mi_scores.shape[1]

    gap = 0
    for factor in range(num_factors):
        mi_f_sorted = np.flip(np.sort(mi_scores[:, factor]))
        gap += (mi_f_sorted[0] - mi_f_sorted[1])
    gap /= num_factors

    return gap


def jemmig(dis_metric_object):
    """
    compute the Joint Entropy Minus Mutual Information Gap (JEMMIG) from a given matrix of Mutual Information
    JEMMIG = H(v_i, z*) - I(v_i, z*) + I(v_i,z°)

    Args:
        dis_metric_object (DisentanglementMetricsInfo): a DisentanglementMetricsInfo object

    Returns:
        float: a single MIG value (good value: low)

    """
    mi_scores = dis_metric_object.mi_scores
    entropies_meta_data = dis_metric_object.get_meta_data_entropies()
    entropies_z = dis_metric_object.get_z_entropies()

    num_factors = mi_scores.shape[1]

    gap = 0
    for factor in range(num_factors):
        mi_f = mi_scores[:, factor]
        z_star = np.argmax(mi_f)

        h_z = entropies_z[z_star]
        h_f = entropies_meta_data[factor]

        mi_f_sorted = np.flip(np.sort(mi_f))
        # use: I(X,Y) = H(X) + H(Y) - H(X,Y) -> H(X,Y) = H(X) + H(Y) - I(X,Y)
        gap += h_z + h_f - 2 * mi_f_sorted[0] + mi_f_sorted[1]
    gap /= num_factors

    return gap


def mig_sup(dis_metric_object):
    """
    compute the MIG-sup from a given matrix of Mutual Information
    MIG-sup = I(z_i, v*) - I(z_i,v°)
    (note: Mutual Information is symmetric)

    Args:
        dis_metric_object (DisentanglementMetricsInfo): a DisentanglementMetricsInfo object

    Returns:
        float: a single MIG value (good value: high)
    """
    mi_scores = dis_metric_object.mi_scores

    num_dims = mi_scores.shape[0]

    gap = 0
    for dim in range(num_dims):
        mi_dim_sorted = np.flip(np.sort(mi_scores[dim]))
        gap += mi_dim_sorted[0] - mi_dim_sorted[1]
    gap /= num_dims

    return gap


def modularity_score(dis_metric_object):
    """
    compute the Modularity Score from a given matrix of Mutual Information
    mod_score = 1 - (sigma_(all factors except v*) I(v_j, z_i)^2) / ( I(v*, z_j)^2 * (M-1) )

    Args:
        dis_metric_object (DisentanglementMetricsInfo): a DisentanglementMetricsInfo object

    Returns:
        float: a single MIG value (good value: high)

    """
    mi_scores = dis_metric_object.mi_scores

    num_dims = mi_scores.shape[0]
    num_factors = mi_scores.shape[1]

    gap = 0
    for dim in range(num_dims):
        mi_dim_sorted = np.flip(np.sort(mi_scores[dim]))
        mi_dim_sorted_squared = np.square(mi_dim_sorted)
        gap += 1 - (np.sum(mi_dim_sorted_squared[1:]) / (mi_dim_sorted_squared[0] * (num_factors - 1)))
    gap /= num_dims

    return gap


def dcimig(dis_metric_object): ...  # (sigma S_i) / (sigma H(v_i))

# PREDICTION-BASED DISENTANGLEMENT METRICS #

# INTERVENTION-BASED DISENTANGLEMENT METRICS #
