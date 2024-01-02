import pandas as pd
from sklearn.manifold import TSNE

from utils.helper import Helper
from utils.visualizations import Visualizations
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


class EvaluateAnomalyDetection():

    def __init__(self, model, path_save, seed=42):
        self._model = model
        self._path_save = path_save + 'evaluation/anomaly/'
        self._seed = seed
        self.names = [
            "Nearest Neighbors",
            # "Linear SVM",
            # "RBF SVM",
            # "Gaussian Process",
            "Decision Tree",
            # "Random Forest",
            # "Neural Net",
            # "AdaBoost",
            # "Naive Bayes",
            # "QDA",
        ]
        self.classifiers = [
            KNeighborsClassifier(3),
            # SVC(kernel="linear", C=0.025, random_state=42),
            # SVC(gamma=2, C=1, random_state=42),
            # GaussianProcessClassifier(1.0 * RBF(1.0), random_state=42),
            DecisionTreeClassifier(max_depth=5, random_state=42),
            # RandomForestClassifier(
            #    max_depth=5, n_estimators=10, max_features=1, random_state=42
            # ),
            # MLPClassifier(alpha=1, max_iter=1000, random_state=42),
            # AdaBoostClassifier(random_state=42),
            # GaussianNB(),
            # QuadraticDiscriminantAnalysis(),
        ]

    def _anomaly_detection(self, X, y, path):
        df = pd.DataFrame()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=self._seed)

        for name, clf in zip(self.names, self.classifiers):
            clf = make_pipeline(StandardScaler(), clf)
            clf.fit(X_train, y_train)
            score = clf.score(X_test, y_test)
            df = pd.concat([df, pd.DataFrame({'method': [str(name)], 'target': [str(name)], 'score': [str(score)]})])
        Helper.generate_paths([path])
        df.to_csv(path + '/anomaly_scores.csv')

    def evaluate(self, datasets):
        for d in datasets.keys():
            name = datasets[d]['name']
            for split in datasets[d]['splits']:
                path = self._path_save + name + '_' + split
                Helper.generate_paths([path])
                z, labels = Helper.get_embedding(self._model, name, split=split, save_path=path)
                embedding_tsne = TSNE().fit_transform(z[:, :, 0])
                Visualizations.plot_embedding(embedding_tsne, labels, path)
                self._anomaly_detection(z[:, :, 0], labels['beat'], path)
