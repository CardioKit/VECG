# visualization
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import animation
import seaborn as sns
import numpy as np
import pandas as pd

from sklearn.manifold import TSNE

class Visualisation():

    def __init__(self, plot_params={}):
        a = 2+3
        
    def plot_timeseries(self, series):
        plt.figure(figsize=(15, 5))
        plt.plot(series)
        
    def plot_reconstruction(self, model, series):
        reconstruction, _, _, _ = model(series)
        residuum = np.abs(series - reconstruction)
        plt.figure(figsize=(15, 5))
        plt.box(False)
        plt.axis('off')
        plt.plot(series[0,:,0], label="Original Signal")
        plt.plot(reconstruction[0,:,0], label="Reconstructed Signal")
        plt.plot(residuum[0,:,0], label="Residuum")
        plt.legend(loc="upper right", fontsize=18)
        
    def scatter_embedding(self, embedding, color, s=3):
        plt.figure(figsize=(15,15))
        plt.scatter(embedding[:, 0], embedding[:, 1], s=s, c=color, label=color, alpha=0.3, edgecolors='none')


class VisualisationOld():

    def __init__(self, x_train, y_train, model):
        self.x_train = x_train
        self.y_train = y_train
        self.model = model
        self.encoded = model.encode(self.x_train)
        self.decoded = model.decode(self.encoded[0])

    def plotToggledFeature(self, sample, feature, plotOrg=True, lowerBound=-100, upperBound=100):

        series = self.x_train[sample, :, 0]
        z_mean = self.encoded[0][sample]
        timeseries = []
        latent_dim = len(z_mean)
        org_dim = self.x_train.shape[1]

        for k in range(lowerBound, upperBound):
            temp = np.copy(z_mean)
            temp[feature] = (temp[feature] + (k / float(upperBound)))
            x_decoded = np.array(self.model.decoder(temp.reshape(1, latent_dim)))
            timeseries.append(x_decoded)

        std = np.std(timeseries, axis=0).reshape(org_dim)
        mean = np.mean(timeseries, axis=0).reshape(org_dim)
        plt.figure(figsize=(15, 10))
        plt.plot(mean, color='blue')
        plt.plot(mean - std, color='orange')
        plt.plot(mean + std, color='orange')
        if plotOrg:
            plt.plot(series.reshape(org_dim), color='green')
        plt.savefig("../results/media/toggledFeature_" + str(feature) + ".svg", dpi=150)

    def plotConfusionMatrix(self, model, x, labelling):

        fig, ax = plt.subplots(figsize=(20, 20))
        plot_confusion_matrix(
            model,
            x,
            labelling,
            cmap=plt.cm.Blues,
            ax=ax,
        )
        plt.savefig("../results/media/confusionMatrix.svg", dpi=150)

    def printConfusionMatrix(self, model, x, y_true, to_latex=False):
        # TODO: Give Credits
        def gen_align(align, size):
            texto = ""
            for i in range(0, size):
                texto += align
            return texto

        def gen_header(size):
            header = " "
            for i in range(0, size):
                header += "& " + str(i) + " "
            header += "\\\ \n"
            return header

        def cm_to_latex(array, align="l"):
            n_cols = len(array[0])
            table = "\\begin{table}\n"
            table += "\caption{Table of Confusion Matrix generated with python}\n"
            table += "\label{cm:python:generated}\n"
            table += "\small\n"
            table += "\\centering\n"
            table += "\\begin{tabular}{@{}" + gen_align(align, n_cols + 1) + "@{}}\n"
            table += gen_header(n_cols)
            for i, linha in enumerate(array):
                table += str(i) + " "
                for val in linha:
                    table += "& " + str(val) + " "
                if i + 1 == n_cols:
                    table += "\\\ \n"
                else:
                    table += "\\\ \n"
            table += "\end{tabular}\n"
            table += "\end{table}"
            print(table)

        y_pred = model.predict(x)
        cm = confusion_matrix(y_pred, y_true)

        if to_latex:
            cm_to_latex(cm)
        else:
            print(cm)

    def plotReconstruction(self, sample):

        series = self.x_train[sample, :, 0]
        reconstructed = self.decoded[sample]

        plt.figure(figsize=(15, 5))
        plt.plot(series)
        plt.plot(reconstructed)
        plt.savefig("../results/media/reconstruction.svg", dpi=150)
        plt.close()

    def plotTSNEmbedding(self, feature=1, minAmountLabels=0):

        x_embedded = TSNE(n_components=2).fit_transform(self.encoded[0])

        unique, counts = np.unique(self.y_train[:, feature], return_counts=True)
        labelCount = {k: v for k, v in dict(zip(unique, counts)).items() if v > minAmountLabels}
        index = np.isin(self.y_train[:, feature], np.array(list(labelCount.keys())))

        #rhythmNames["Acronym Name"] = rhythmNames["Acronym Name"].str.replace(' ', '')
        #temp = pd.DataFrame(
        #    self.y_train[index, feature],
        #    columns=['Acronym Name'],
        #).merge(rhythmNames, on='Acronym Name', how='left')

        data = [x_embedded[index, 0], x_embedded[index, 1], self.y_train[:, feature]]
        df = pd.DataFrame(data).T
        df.columns = ['First Dimension', 'Second Dimension', 'Description']

        sns.set(rc={'figure.figsize': (15.0, 15.0)})
        sns.color_palette("viridis", as_cmap=True)
        sns.scatterplot(data=df, x="First Dimension", y="Second Dimension", hue="Description",
                        s=15)  # , palette='Paired')
        plt.savefig("../results/media/embedding.svg", dpi=150)
        plt.close()

    def categoricalToColor(self, labelling):
        '''
        Transforms a categorical value into a numeric representation usable as color
        '''
        colors = pd.Categorical(labelling).codes
        return colors

    def plotElectrocardiogramSample(self, sample, channels):
        k = len(channels)
        fig, axs = plt.subplots(k, 1, sharex=True, figsize=(8, 12))

        for idx, value in enumerate(channels):
            # TODO Think about how to include also the other leads
            axs[idx].plot(self.x_train[sample, :, 0])
            axs[idx].axis('off')
            axs[idx].set_title(str(value), loc='left')

    def generateAnimation(self, feature, name='../results/media/animation_tsne.mp4', progress=True, frames=360, interval=1):
        def init():
            ax.view_init(elev=10., azim=0)
            return [scat]

        def animate(i):
            if progress:
                print(i)
            ax.view_init(elev=10., azim=i)
            return [scat]

        color = self.categoricalToColor(self.y_train[:, feature])
        x_embedded = TSNE(n_components=3).fit_transform(self.encoded[0])

        fig = plt.figure(figsize=(20, 20))
        ax = Axes3D(fig)

        scat = ax.scatter(x_embedded[:, 0], x_embedded[:, 1], x_embedded[:, 2], c=color, s=12, cmap='tab20')
        anim = animation.FuncAnimation(fig, animate, init_func=init, frames=frames, interval=interval, blit=True)

        anim.save(name, fps=30, extra_args=['-vcodec', 'libx264'])
