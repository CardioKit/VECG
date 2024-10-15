import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import tensorflow as tf
import seaborn as sns
from neurokit2.signal import signal_smooth
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import f1_score

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
    def eval_dimensions(ld, model, dimension, path, dpi, df=None, l_bound=-10.0, u_bound=10.0, num_samples=1000):

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
        fig.savefig(path, dpi=dpi)

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
    def plot_embedding_slice(df, dim_x, dim_y, hue, title_legend, path, dpi):
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
        fig.savefig(path, dpi=dpi)

    @staticmethod
    def plot_confustion_matrix_(X_train, X_test, y_train, y_test, predictor, path, dpi, cmap='Greens'):
        predictor.fit(X_train.fillna(0), y_train)
        predictions = predictor.predict(X_test.fillna(0))
        cm = confusion_matrix(y_test, predictions, labels=predictor.classes_)
        fig = plt.figure(figsize=(15, 15))
        fig.tight_layout()
        disp = ConfusionMatrixDisplay(confusion_matrix=np.round(cm / np.sum(cm, axis=0), 2), display_labels=predictor.classes_)
        disp.plot(cmap=cmap)
        plt.show()
        fig.savefig(path, dpi=dpi)
        return cm

    @staticmethod
    def plot_along_axis(dim, feature, ld, x, path, model, dpi):
        plt.figure(figsize=(15, 5))
        for k in x:
            embedding = np.zeros(ld).astype(np.float32)
            embedding[dim] = k
            embedding = np.reshape(embedding, (1, ld))
            decoded = signal_smooth(np.reshape(model.decode(embedding), 500))
            plt.plot(decoded, label=str(k))
            plt.title('Dimension ' + str(dim) + ': ' + feature)
        plt.legend(title='Value', frameon=False)
        plt.savefig(path + 'reconstruction_toggle_' + str(dim) + '.png', dpi=dpi, bbox_inches='tight')

    @staticmethod
    def reconstruct(dim, x, model, ld):
        embedding = np.zeros(ld).astype(np.float32)
        embedding[dim] = x
        embedding = np.reshape(embedding, (1, ld))
        decoded = signal_smooth(np.reshape(model.decode(embedding), 500))
        plt.figure(figsize=(15, 5))
        plt.plot(decoded)

    @staticmethod
    def plot_confustion_matrix(X_train, X_test, y_train, y_test, predictor, path, dpi, normalize=False, cmap='Greens'):
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
        disp.figure_.savefig(path, dpi=dpi)
        return cm, predictor.classes_, predictions

    @staticmethod
    def print_metrics_binary(cm, y_test, predictions):
        print('Accuracy:   \t', np.trace(cm) / np.sum(cm), '\n')
        print('Sensitivity:\t', cm[0, 0] / (cm[0, 0] + cm[1, 0]), '\n')
        print('Specificity:\t', cm[1, 1] / (cm[1, 1] + cm[0, 1]), '\n')
        print('F1 Score:\t', f1_score(y_test, predictions), '\n')

    @staticmethod
    def print_metrics_multiclass(cm, y_test, predictions):
        print('Accuracy:   \t', np.trace(cm) / np.sum(cm), '\n')
        print('Weighted F1 Score:   \t', f1_score(y_test, predictions, average='weighted'), '\n')

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
    def plot_axis_relation(interpretation, ld, save_path, dpi):
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
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight')

    @staticmethod
    def print_axis_interpretation(interpretation, depth=4):
        columns = ['Dimension'] + ['Feature_' + str(k) for k in range(0, depth)]
        df = pd.DataFrame(columns=columns)
        for k in interpretation.items():
            row = [k[0][4:]]
            for j in range(0, 4):
                row.append(str(k[1]['Features'][j]) + ': ' + str(np.round(k[1]['Scores'][j], 5)))
            temp = pd.DataFrame(row, index=columns).transpose()
            df = pd.concat([df, temp])
        return df

    @staticmethod
    def decode_and_smooth(dim_values, ld, model, dim):
        # Precompute all decoded signals for a range of values in a given dimension
        embeddings = np.zeros((len(dim_values), ld))
        for i, k in enumerate(dim_values):
            embeddings[i, dim] = k
        decoded_signals = [
            signal_smooth(np.array(model.decode(tf.cast(embedding.reshape(1, ld), tf.float32))).reshape(500)) for
            embedding in embeddings]
        return decoded_signals

    @staticmethod
    def plot_with_facetgrid(ld, x_values, path, model, interpretation, dpi, dimensions, palette='viridis'):
        all_data = []
        for dim in range(ld):
            dim_values = Visualizations.decode_and_smooth(x_values, ld, model, dim)
            for i, value in enumerate(x_values):
                for point in dim_values[i]:
                    all_data.append({'Dimension': dim, 'Value': value, 'Signal': point})
    
        data = pd.DataFrame(all_data)
        data.reset_index(inplace=True)
        data['index'] = data['index'] % 500
        data = data[data.Dimension.isin(dimensions)]
        g = sns.FacetGrid(data, col='Dimension', col_wrap=1, sharex=True, aspect=2, height=3)
        g.map_dataframe(sns.lineplot, x='index', y='Signal', hue='Value', palette=palette)
        
        for ax, dim in zip(g.axes.flatten(), dimensions):
            feature = interpretation['Dim ' + str(dim)]['Features']
            rater = interpretation['Dim ' + str(dim)]['Rater']
    
            ax.set_xlabel("Time [ms]")
            ax.set_ylabel("")
            ax.set_yticks([0, 0.5, 1])
            description_text = 'Analysis \n\u2022 ' + feature[0] + '\n\n' + 'Expert'
            for k in rater:
                description_text += '\n\u2022 ' + k
            ax.text(1.0, 0.55, description_text, horizontalalignment='left', verticalalignment='center',
                    transform=ax.transAxes, bbox=dict(facecolor='white', edgecolor='white', alpha=0.5, boxstyle='round,pad=1'))
        g.add_legend(bbox_to_anchor=(0.44, 1.05), loc='upper center', ncol=5, title="Value of the dimension")
        plt.tight_layout(pad=0)
        plt.savefig(path + 'reconstruction_grid.png', dpi=dpi, bbox_inches='tight')
    
    @staticmethod
    def plot_scatter(embedding, col1, col2, vars1, vars2, name1, name2, DIP,
                     path_save='./analysis/media/embedding_synthetic.png', palette='viridis'):
        
        df_melted = pd.melt(embedding, id_vars=[col1, col2],
                                value_vars=[vars1, vars2],
                                var_name='Wave Characteristic', value_name='Color_')

        df_melted['Wave Characteristic'] = df_melted['Wave Characteristic'].replace('t_height', name1).replace('p_height', name2)

        # Convert 8.8 cm to inches for width
        width_in_inches = 8.8 / 2.54
        # Assume a height in inches that seems reasonable (you might need to adjust this)
        height_in_inches = width_in_inches * (3/4)  # Example aspect ratio
    
        # Create a figure with the desired size
        plt.figure(figsize=(width_in_inches, height_in_inches))
        g = sns.FacetGrid(df_melted, col='Wave Characteristic', height=5, aspect=1)

        # Map the scatterplot
        g.map(sns.scatterplot, col1, col2, 'Color_', s=5, palette=palette)

        g.set_axis_labels("", 'Dimension ' + str(col2))

        n_cols = g.axes.shape[0]

        for i, ax in enumerate(g.axes.flat):
            if i < len(g.axes.flat) - n_cols:
                ax.set_xlabel('')

        g.fig.text(0.5, 0.04, 'Dimension ' + str(col1), ha='center')
    
        #g.fig.legend(markerscale=7, loc='upper center', bbox_to_anchor=(0.5, 1.04), ncol=5, borderaxespad=0., edgecolor='white')
        plt.legend(markerscale=5, frameon=False, loc='upper center', bbox_to_anchor=(0., 1.25), ncol=5, edgecolor='white')
        # plt.tight_layout()
        fig = g.figure
        fig.savefig(path_save, dpi=DIP, bbox_inches='tight')
