import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from neurokit2.signal import signal_smooth
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


class Visualizations:

    @staticmethod
    def plot_embedding(embedding, labels):
        NotImplementedError

    @staticmethod
    def pair_plot(embedding, labels):
        NotImplementedError

    @staticmethod
    def eval_reconstruction(X, reconstruction, indices, path_eval, titles=None, xlabel=None, ylabel=None):
        """
        Plots original and reconstructed data for given indices.

        Parameters:
            X (numpy.ndarray): Original data.
            reconstruction (numpy.ndarray): Reconstructed data.
            indices (list): List of indices to plot.
            titles (list, optional): Titles for each subplot.
            xlabel (str, optional): Label for the x-axis.
            ylabel (str, optional): Label for the y-axis.

        Returns:
            None
        """
        assert len(indices) == 4
        num_rows = 2
        num_cols = 2

        fig, ax = plt.subplots(num_rows, num_cols, sharex=True, sharey=True, figsize=(15, 5))

        for i, idx in enumerate(indices):
            row = i // num_cols
            col = i % num_cols
            ax[row, col].plot(X[idx], label='Original')
            ax[row, col].plot(reconstruction[idx], label='Reconstruction')

            if titles:
                ax[row, col].set_title(titles[i])

            ax[row, col].legend(loc='upper left')

        if xlabel:
            for a in ax[-1, :]:
                a.set_xlabel(xlabel)
        if ylabel:
            for a in ax[:, 0]:
                a.set_ylabel(ylabel)

        plt.tight_layout()
        plt.savefig(path_eval + 'reconstruction.png')
        plt.close()

    @staticmethod
    def eval_dimensions(ld, model, dimension, path, df=None, l_bound=-10.0, u_bound=10.0, num_samples=1000):

        mean_values = np.zeros(ld).astype(np.float32) #, std_values = np.mean(df.iloc[:, :ld], axis=0), np.std(df.iloc[:, :ld], axis=0)
        result_matrix = np.tile(mean_values, (num_samples, 1))
        result_matrix[:, dimension] = np.linspace(l_bound, u_bound, num_samples)
        X = model.decode(result_matrix)

        M = np.zeros((X.shape))
        for k, _ in enumerate(X):
            M[k] = signal_smooth(X[k].numpy())

        mean = np.mean(M, axis=0)
        std = np.std(M, axis=0)
        fig = plt.figure(figsize=(15, 5))
        fig.tight_layout()
        plt.plot(range(0, len(mean)), mean, 'k-')
        plt.fill_between(range(0, len(mean)), mean - std, mean + std)
        plt.title("ECG reconstruction by toggling dimension " + str(dimension) + ".")
        fig.savefig(path, dpi=300)

    @staticmethod
    def plot_trainings_process(train_progress, metrics):
        fig = plt.figure(figsize=(10, 5))
        fig.tight_layout()
        for k in metrics:
            ax = sns.lineplot(train_progress, x='epoch', y=k, legend=False, label=k)
            ax.set_yscale("log")
        plt.legend()
        plt.show()

    @staticmethod
    def plot_variations(df, ld, model, dimension=0, num_rows=1000):
        mean_values = np.mean(df.iloc[:, :ld], axis=0)
        std_values = np.std(df.iloc[:, :ld], axis=0)
        result_matrix = np.tile(mean_values, (num_rows, 1))
        result_matrix[:, dimension] = np.linspace(-10.0, 10.0, num_rows)
        X = model.decode(result_matrix)

        reconstruct = pd.DataFrame()
        reconstruct['values'] = X.numpy().flatten()

        original_array = list(range(0, 500))
        desired_length = len(reconstruct)
        repeating_array = [original_array[i % len(original_array)] for i in range(desired_length)]

        reconstruct['timestamp'] = repeating_array
        plt.figure(figsize=(15, 5))
        sns.lineplot(data=reconstruct, x="timestamp", y="values")

    @staticmethod
    def plot_embedding_slice(df, dim_x, dim_y, hue, title_legend, path):
        fig = plt.figure(figsize=(10, 10))
        fig.tight_layout()
        ax = sns.scatterplot(
            data=df, x=dim_x, y=dim_y, hue=hue,
        )
        ax.set(
            xlabel='Dimension ' + str(dim_x),
            ylabel='Dimension ' + str(dim_y),
            title="Slice through the embedding space.",
        )
        plt.legend(title=title_legend, frameon=False)
        plt.tight_layout()
        plt.show()
        fig.savefig(path, dpi=300)

    @staticmethod
    def plot_confustion_matrix(X_train, X_test, y_train, y_test, predictor, path, cmap='Greens'):
        predictor.fit(X_train.fillna(0), y_train)
        predictions = predictor.predict(X_test.fillna(0))
        cm = confusion_matrix(y_test, predictions, labels=predictor.classes_)
        fig = plt.figure(figsize=(15, 15))
        fig.tight_layout()
        disp = ConfusionMatrixDisplay(confusion_matrix=np.round(cm / np.sum(cm, axis=0), 2), display_labels=predictor.classes_)
        disp.plot(cmap=cmap)
        plt.show()
        fig.savefig(path, dpi=300)
        return cm

    @staticmethod
    def plot_along_axis(dim, feature, ld, x, path, model):
        plt.figure(figsize=(15, 5))
        for k in x:
            embedding = np.zeros(ld).astype(np.float32)
            embedding[dim] = k
            embedding = np.reshape(embedding, (1, ld))
            decoded = signal_smooth(np.reshape(model.decode(embedding), 500))
            plt.plot(decoded, label=str(k))
            plt.title('Dimension ' + str(dim) + ': ' + feature)
        plt.legend(title='Value', frameon=False)
        plt.savefig(path + 'reconstruction_toggle_' + str(dim) + '.png', dpi=300, bbox_inches='tight')

    @staticmethod
    def reconstruct(dim, x, model, ld):
        embedding = np.zeros(ld).astype(np.float32)
        embedding[dim] = x
        embedding = np.reshape(embedding, (1, ld))
        decoded = signal_smooth(np.reshape(model.decode(embedding), 500))
        plt.figure(figsize=(15, 5))
        plt.plot(decoded)

    @staticmethod
    def plot_confustion_matrix(X_train, X_test, y_train, y_test, predictor, path, normalize=False, cmap='Greens'):
        predictor.fit(X_train.fillna(0), y_train)
        predictions = predictor.predict(X_test.fillna(0))
        cm = confusion_matrix(y_test, predictions, labels=predictor.classes_)
        fig = plt.figure(figsize=(15, 15))
        fig.tight_layout()
        if normalize:
            disp = ConfusionMatrixDisplay(confusion_matrix=np.round(cm / np.sum(cm, axis=1), 2),
                                          display_labels=predictor.classes_)
        else:
            disp = ConfusionMatrixDisplay(confusion_matrix=np.round(cm, 2), display_labels=predictor.classes_)
        disp.plot(cmap=cmap)
        plt.show()
        disp.figure_.savefig(path, dpi=300)
        return cm, predictor.classes_

    @staticmethod
    def print_metrics_binary(cm):
        print('Accuracy:   \t', np.trace(cm) / np.sum(cm), '\n')
        print('Sensitivity:\t', cm[0, 0] / (cm[0, 0] + cm[1, 0]), '\n')
        print('Specificity:\t', cm[1, 1] / (cm[1, 1] + cm[0, 1]), '\n')

    @staticmethod
    def print_metrics_multiclass(cm):
        print('Accuracy:   \t', np.trace(cm) / np.sum(cm), '\n')

    @staticmethod
    def pandas_to_latex(df):
        return df.round(2).sort_values(
            by=['latent_dim', 'alpha', 'beta', 'gamma'],
        )[[
            'latent_dim', 'alpha', 'beta', 'gamma', 'loss', 'recon', 'mi', 'tc', 'dw_kl', 'MIG'
        ]].to_latex(
            index=False
        ).replace(
            '\n', ''
        ).replace(
            '.000000', ''
        ).replace(
            '0000', ''
        )

    @staticmethod
    def plot_axis_relation(interpretation, ld, save_path, DPI=300):
        a = pd.DataFrame()
        ind = 0
        for k in interpretation:
            b = pd.DataFrame(interpretation[k]['Scores']).transpose()
            b.columns = interpretation[k]['Features']
            b = b.loc[:, ~b.columns.duplicated()].copy()
            a = pd.concat([a, b])
            ind = ind + 1
        a.index = range(0, ld)
        fig = plt.figure(figsize=(5, 5))
        ax = sns.heatmap(a.transpose(), cmap="crest")
        ax.set(xlabel="Dimension", ylabel="Feature")
        fig.savefig(save_path, dpi=DPI, bbox_inches='tight')

    @staticmethod
    def print_axis_interpretation(interpretation):
        print('Dimension', '\t|', 'Features and Scores (ordered by score)')
        print('____________________________________________________________________________')
        for k in interpretation.items():
            string_build = str(k[0][4:]) + '\t\t|'
            for j in range(0, 4):
                string_build += str(k[1]['Features'][j]) + ': ' + str(np.round(k[1]['Scores'][j], 5)) + '\t|'
            print(string_build)
