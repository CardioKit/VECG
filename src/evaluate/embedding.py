from utils.helper import Helper


class Embedding:

    def __init__(self, model, path_save):
        self._model = model
        self._path_save = path_save + 'embedding/'

    def evaluate_dataset(self, dataset):
        name = dataset['name']
        for split in dataset['splits']:
            path = self._path_save + name + '/' + split
            Helper.generate_paths([path])
            _, _ = Helper.get_embedding(self._model, name, split, save_path=path)
