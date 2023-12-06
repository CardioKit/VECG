import tensorflow as tf

class Utils:

    @staticmethod
    def data_generator(dataset):
        iterator = iter(dataset)
        while True:
            try:
                batch = next(iterator)
                yield batch['ecg']['I']
            except StopIteration:
                iterator = iter(dataset)

    @staticmethod
    def get_samples(dataset, n, label=None):
        k = None
        for example in dataset.take(1):
            k = example
        return (k['ecg']['I'][0:n], k[label][0:n]) if label else k['ecg']['I'][0:n]

    @staticmethod
    def scheduler(epoch, lr):
        if epoch < 20:
            return lr
        else:
            return lr * tf.math.exp(-0.5)
