import os, yaml, codecs, json
import tensorflow as tf


class Utils:

    @staticmethod
    def generate_paths(paths):
        for path in paths:
            try:
                os.makedirs(path, exist_ok=True)
                print(f"Created directory: {path}")
            except FileExistsError:
                print(f"Directory already exists: {path}")

    @staticmethod
    def data_generator(dataset, method='continue'):
        iterator = iter(dataset)
        while True:
            try:
                batch = next(iterator)
                yield batch['ecg']['I']
            except StopIteration:
                if method == 'continue':
                    iterator = iter(dataset)
                elif method == 'stop':
                    pass

    @staticmethod
    def get_sample(dataset, n, label=None):
        k = None
        for example in dataset.take(1):
            k = example
        return (k['ecg']['I'][n:(n + 1)], k[label][n:(n + 1)]) if label else k['ecg']['I'][n:(n + 1)]

    @staticmethod
    def scheduler(epoch, lr):
        if epoch < 20:
            return lr
        else:
            return lr * tf.math.exp(-0.5)

    @staticmethod
    def load_yaml_file(path):
        with open(path, "r") as stream:
            try:
                return yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)

    @staticmethod
    def write_json_file(file, filepath):
        with codecs.open(filepath, 'w', 'utf8') as f:
            f.write(json.dumps(file, sort_keys=True, ensure_ascii=False))
