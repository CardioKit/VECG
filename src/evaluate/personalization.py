import tensorflow as tf

from utils.helper import Helper


class EvaluatePersonalization():
    def __init__(self, model, path_save):
        self._model = model
        self._path_save = path_save + 'evaluation/personlization/'

    def fine_tune_evaluate(self, dataset, split, epochs):
        model_fine_tune = tf.keras.clone_model(self._model)
        model_fine_tune.fit(
            Helper.data_generator(dataset),
            len(dataset),
        )
        z_mean, _, _ = model_fine_tune.encode(dataset)


'''
    data = tfds.load('synth')
    ds = data['train'].shuffle(128).batch(50000)
    iterator = iter(ds)
    batch = next(iterator)
    X = batch['ecg']['I']
    z_mean, z_log_var, z = dvae.encode(X)

    datapre = {
        'meta_data': np.array(
            [batch['p_height'],
             batch['t_height']]
        ).reshape(-1, 2),
        'z': np.array(z),
        'discrete_features': [False, False]
    }
    datapre = dotdict(datapre)
    sim_measure = SimilarityMeasure(datapre, linear_regressor)
    dis_metric = DisentanglementMetricsInfo(sim_measure, mutual_information_gap, None)
    print(dis_metric.get_meta_data_entropies())
'''