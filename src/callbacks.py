import tensorflow as tf
import pandas as pd
from sklearn.manifold import TSNE
import io
import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
import seaborn as sns
from matplotlib import pyplot as plt

class KLCoefficientScheduler(tf.keras.callbacks.Callback):
    """KL coefficient scheduler which sets the KL weight according to schedule.

    Arguments:
      coefficient: a function that takes an epoch index
          (integer, indexed from 0) and current kl coefficient
          as inputs and returns a new kl coefficient as output (float).
    """
    def frange_cycle_cosine(self, start, stop, n_epoch, n_cycle=4, ratio=0.5, shift=0.0):
        import math
        L = np.ones(n_epoch)
        period = n_epoch/n_cycle
        step = (stop-start)/(period*ratio)

        for c in range(n_cycle):
            v , i = start , 0
            while v <= stop:
                L[int(i+c*period)] = (math.cos(v*math.pi))*0.5 + 0.5
                v += step
                i += 1
        return L
    
    def frange_cosine(self, n_epoch):
        L = 0.5*(np.sin(np.linspace(0, 4*np.pi, n_epoch) - 0.25*np.pi)*(-1.0)).clip(0)
        return L
    
    def zero_cosine(self, n_epoch):
        L = np.zeros(n_epoch)
        return L
    
    def __init__(self, epochs, start=0.0, stop=1.0, cycles=2, ratio=0.5, shift=0.0):
        super(KLCoefficientScheduler, self).__init__()
        self.beta_cycle = self.frange_cycle_cosine(start, stop, epochs, cycles, ratio, shift)
        #self.beta_cycle = self.frange_cycle_cosine(start, stop, epochs, cycles, ratio=ratio)
    
    @property
    def betacycle(self):
        return self.beta_cycle
    
    def plot_beta_cycle(self):
        from matplotlib import pyplot as plt
        
        plt.figure(figsize=(12,5))
        plt.plot(self.betacycle)
        plt.xlabel("Epoch")
        plt.ylabel("β")
        plt.show()

    def on_epoch_begin(self, epoch, logs=None):
        n = len(self.beta_cycle)
        index = epoch % n
        self.model.beta = self.beta_cycle[epoch]
        
class LatentVectorSpaceSnapshot(tf.keras.callbacks.Callback):

    def __init__(self, data, labelling, log_dir, period=10):
    
        super(LatentVectorSpaceSnapshot, self).__init__()
        self.data = data
        self.labelling = labelling
        self.period = period
        self.summary_writer = tf.summary.create_file_writer(log_dir)

    def plot_to_image(self, figure):
        """Saves the matplotlib plot specified by 'figure', converts it to a PNG image and returns it."""

        # Save the plot to a PNG at the specified path
        #figure.savefig(filename, format='png', bbox_inches='tight', dpi=150)
        # Save the plot to a PNG in memory.
        buf = io.BytesIO()
        figure.savefig(buf, format='png', bbox_inches='tight', dpi=200)
        #figure.show()  # log the figure
        plt.close(figure.fig)  # do not log the figure
        buf.seek(0)
        # Convert PNG buffer to TF image
        image = tf.image.decode_png(buf.getvalue(), channels=4)
        # Add the batch dimension
        image = tf.expand_dims(image, 0)

        return image
        
    def on_epoch_end(self, epoch, logs=None):
        
        if epoch % self.period == 0:
            with self.summary_writer.as_default():
                encoded = self.model.encoder(self.data)
                
                x_tsne = TSNE(n_components=3).fit_transform(encoded[0])
                df = pd.DataFrame(np.array(x_tsne))
                df['labelling'] = self.labelling
                
                figure = sns.pairplot(df, hue="labelling")
                tsne_image = self.plot_to_image(figure)
                tf.summary.image(f'pairplot/', tsne_image, step=epoch)
                
                self.summary_writer.flush()


class ReconstructionPlot(tf.keras.callbacks.Callback):

    def __init__(self, data, log_dir, period=10):
    
        super(ReconstructionPlot, self).__init__()
        self.data = data
        self.period = period
        self.summary_writer = tf.summary.create_file_writer(log_dir)

    def plot_to_image(self, figure):
        """Saves the matplotlib plot specified by 'figure', converts it to a PNG image and returns it."""

        # Save the plot to a PNG at the specified path
        #figure.savefig(filename, format='png', bbox_inches='tight', dpi=150)
        # Save the plot to a PNG in memory.
        buf = io.BytesIO()
        figure.figure.savefig(buf, format='png', bbox_inches='tight', dpi=200)
        #figure.show()  # log the figure
        plt.close(figure.figure)  # do not log the figure
        buf.seek(0)
        # Convert PNG buffer to TF image
        image = tf.image.decode_png(buf.getvalue(), channels=4)
        # Add the batch dimension
        image = tf.expand_dims(image, 0)

        return image
        
    def on_epoch_end(self, epoch, logs=None):
        
        if epoch % self.period == 0:
            with self.summary_writer.as_default():
            
                encoded = self.model.encoder(self.data)
                decoded = self.model.decoder(encoded[0])
                
                len_data = len(self.data)
                len_signal = len(self.data[0])
                
                df_org = pd.DataFrame(self.data.reshape(len_signal, len_data), columns=['signal'])
                df_rec = pd.DataFrame(np.array(decoded).reshape(len_signal), columns=['signal'])
                df_err = (df_org-df_rec).abs()

                df_org['type'] = 'original'
                df_org['timepoint'] = range(0,len_signal)
                df_rec['type'] = 'reconstruction'
                df_rec['timepoint'] = range(0,len_signal)
                df_err['type'] = 'error'
                df_err['timepoint'] = range(0,len_signal)

                df = df_org.append(df_rec).append(df_err)
                df.index = range(0, len(df))
                
                figure = sns.lineplot(x="timepoint", y="signal", hue="type", data=df)
                tsne_image = self.plot_to_image(figure)
                tf.summary.image(f'reconstruction/', tsne_image, step=epoch)
                
                self.summary_writer.flush()
