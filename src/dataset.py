import pandas as pd
import numpy as np
import wfdb
import ast
import glob
import pickle
import gzip
import os
import neurokit2 as nk
import sklearn.preprocessing as preprocessing


class Dataset:
    DATASET_NAME = ''
    SAMPLING_RATE = 0

    def __init__(self):
        self.X = None
        self.Y = None

    def loadData(self, name, path='../temp/'):
        import numpy as np
        '''
        Loads data and associated labelling from hard disc
        '''
        try:
            X = np.load(path + 'x_' + name + '.npy', allow_pickle=True)
            y = np.load(path + 'y_' + name + '.npy', allow_pickle=True)
        except Exception as e:
            print(e)
        return X, y

    def saveData(self, name, path='../temp/'):
        import numpy as np
        '''
        Writes data and associated labelling to hard disc
        '''
        try:
            np.save(path + 'x_' + name + '.npy', self.X)
            np.save(path + 'y_' + name + '.npy', self.Y)
        except Exception as e:
            print(e)
        return True

    def dataset_info(self):
        print("Name:", self.DATASET_NAME, ",",
              "Shape:", self.X.shape, ",",
              "Sampling Rate:", self.SAMPLING_RATE)


class DatasetZheng(Dataset):
    DATASET_NAME = 'zheng'
    SAMPLING_RATE = 500

    def __init__(self, path, preloaded=False, save=True, folder='../temp/', name='zheng', channel=['I']):
        super().__init__()
        self.path = path
        self.folder = folder
        self.save = save
        self.name = name
        self.preloaded = preloaded
        self.channel = channel
        self.columns = pd.read_excel(path + 'Diagnostics.xlsx', nrows=0).columns

        if self.preloaded:
            self.X, self.Y = self.loadData(self.name)
        else:
            self.metadata = pd.read_excel(path + 'Diagnostics.xlsx')
            self.X = np.zeros((len(self.metadata), 5000))
            self.Y = np.array(self.metadata)
            self.loadDataset()
            self.saveData(self.name)

    def loadDataset(self):
        for idx, k in enumerate(self.path + self.metadata.FileName + '.csv'):
            try:
                self.X[idx] = np.genfromtxt(
                    k, delimiter=',', names=True, usecols=(self.channel), encoding='utf-8-sig',
                )
            except Exception as e:
                print(e, idx)


class DatasetGlucose(Dataset):
    DATASET_NAME = 'glucose'
    SAMPLING_RATE = 500

    def __init__(self, path, preloaded=False, save=True, folder='../temp/', name='zheng', channel=['I']):
        super().__init__()
        self.path = path
        self.folder = folder
        self.save = save
        self.name = name
        self.preloaded = preloaded
        self.channel = channel
        self.columns = pd.read_excel(path + 'Diagnostics.xlsx', nrows=0).columns

        if self.preloaded:
            self.X, self.Y = self.loadData(self.name)
        else:
            self.metadata = pd.read_excel(path + 'Diagnostics.xlsx')
            self.X = np.zeros((len(self.metadata), 5000))
            self.Y = np.array(self.metadata)
            self.loadDataset()
            self.saveData(self.name)

    def loadDataset(self):
        for idx, k in enumerate(self.path + self.metadata.FileName + '.csv'):
            try:
                self.X[idx] = np.genfromtxt(
                    k, delimiter=',', names=True, usecols=(self.channel), encoding='utf-8-sig',
                )
            except Exception as e:
                print(e, idx)


class DatasetSimulated(Dataset):
    import glob

    DATASET_NAME = 'simulated'
    SAMPLING_RATE = 500

    def __init__(self, path, preloaded=False, save=True, folder='../temp/', name='simulated', channel=['I']):
        super().__init__()
        self.path = path
        self.folder = folder
        self.save = save
        self.name = name
        self.preloaded = preloaded
        self.channel = channel
        self.files = glob.glob(path + "*.csv")

        if self.preloaded:
            self.X, self.Y = self.loadData(self.name)
        else:
            self.X = np.zeros((len(self.files), 15000))
            self.Y = np.loadtxt(open(path + "config.txt", "rb"), delimiter=",", skiprows=0)
            self.loadDataset()
            self.saveData(self.name)

    def loadDataset(self):
        import scipy
        for idx, k in enumerate(self.files):
            try:
                # mat = scipy.io.loadmat(k)['s'][0:15000]
                # mat = mat.reshape(len(mat))
                mat = np.genfromtxt(k)
                self.X[idx] = mat[0:15000]
            except Exception as e:
                print(e, idx)


class DatasetPTB(Dataset):
    DATASET_NAME = 'ptb'
    SAMPLING_RATE = 500

    def __init__(self, path, preloaded=False, save=True, folder='../temp/', name='ptb'):
        super().__init__()
        self.path = path
        self.folder = folder
        self.save = save
        self.name = name
        self.preloaded = preloaded

        if self.preloaded:
            self.X, self.Y = self.loadData('ptb')
        else:
            # load and convert annotation data
            self.metadata = pd.read_csv(self.path + 'ptbxl_database.csv', index_col='ecg_id')
            self.metadata.scp_codes = self.metadata.scp_codes.apply(lambda x: ast.literal_eval(x))
            self.Y = self.metadata
            # Load raw signal data
            self.X = self.loadDataset(self.Y, self.SAMPLING_RATE, self.path)[:, :, 0]

            # Load scp_statements.csv for diagnostic aggregation
            self.Y['diagnostic_superclass'] = self.Y.scp_codes.apply(self.aggregateDiagnostic)
            self.Y = np.array(self.Y)
            self.saveData(self.name)

    def loadDataset(self, df, sampling_rate, path):
        if sampling_rate == 100:
            data = [wfdb.rdsamp(path + f) for f in df.filename_lr]
        else:
            data = [wfdb.rdsamp(path + f) for f in df.filename_hr]
        data = np.array([signal for signal, meta in data])
        return data

    def aggregateDiagnostic(self, y_dic):
        tmp = []
        agg_df = pd.read_csv(self.path + 'scp_statements.csv', index_col=0)
        agg_df = agg_df[agg_df.diagnostic == 1]
        for key in y_dic.keys():
            if key in agg_df.index:
                tmp.append(agg_df.loc[key].diagnostic_class)
        return list(set(tmp))


class DatasetMobile(Dataset):
    DATASET_NAME = 'mobile'
    SAMPLING_RATE = 512

    def __init__(self, path, preloaded=False, save=True, folder='../temp/', name='mobile'):
        super().__init__()
        self.path = path
        self.folder = folder
        self.save = save
        self.name = name
        self.preloaded = preloaded
        self.filenames = glob.glob(path + "*.csv")
        if self.preloaded:
            self.X, self.Y = self.loadData(self.name)
        else:
            self.X = []
            self.Y = []
            self.loadDataset()
            self.Y = np.asarray(self.Y)
            self.X = np.asarray(self.X)
            self.saveData(self.name)

    def loadDataset(self):
        for idx, k in enumerate(self.filenames):
            try:
                temp = pd.read_csv(k)
                if len(np.array(temp.iloc[9:, 0])) == 15360:
                    self.Y.append(np.array(temp.iloc[0:8, 1]))
                    self.X.append(np.array(temp.iloc[9:, 0], dtype=float))
                    '''
                    self.X[idx] = np.genfromtxt(
                        k,
                        delimiter=',',
                        skip_header=11,
                        names=True,
                        usecols=('Unit'),
                        encoding='utf-8-sig',
                    )
                    '''
            except Exception as e:
                print(k)
                print(e, idx)


class DatasetIcentia(Dataset):
    DATASET_NAME = 'icentia'
    SAMPLING_RATE = 250

    def __init__(self, path, name='icentia', channel=['I']):
        super().__init__()
        self.path = path
        self.name = name
        self.channel = channel
        self.X = np.array([])
        self.y = np.array([])

    def loadDatasetBatch(self, sample_id, save=False, save_path='../temp/'):

        X_temp = []
        y_temp = []

        filename = os.path.join(self.path, ("%05d" % sample_id) + "_batched_lbls.pkl.gz")
        filename_ecg = os.path.join(self.path, ("%05d" % sample_id) + "_batched.pkl.gz")

        # TODO: properly handle missing files
        if (not os.path.isfile(filename)):
            print(sample_id, filename, "##### File missing")

        segments = pickle.load(gzip.open(filename))
        ecg = pickle.load(gzip.open(filename_ecg))
        np.random.seed(42)
        indices = np.random.choice(50, 3)
        # for segment_id, segment_labels in enumerate(segments[0:2]):
        for segment_id in indices:
            segment_labels = segments[segment_id]
            beat_val = []
            beat_label = []
            rhythm_val = []
            rhythm_label = []
            for k in range(len(segment_labels['btype'])):
                temp = segment_labels['btype'][k]
                beat_val.append(temp)
                beat_label.append(np.full(len(temp), k))
            for k in range(len(segment_labels['rtype'])):
                temp = segment_labels['rtype'][k]
                rhythm_val.append(temp)
                rhythm_label.append(np.full(len(temp), k))

            beat_val = np.concatenate(beat_val) * 2
            beat_label = np.concatenate(beat_label)

            rhythm_val = np.concatenate(rhythm_val) * 2
            rhythm_label = np.concatenate(rhythm_label)

            ecg_clean = nk.ecg_clean(ecg[segment_id], sampling_rate=self.SAMPLING_RATE)

            ecg_clean = nk.signal.signal_resample(
                ecg_clean,
                desired_length=2 * len(ecg_clean),
                sampling_rate=self.SAMPLING_RATE
            )
            for i, j in enumerate(beat_val):
                if (j < 250) or ((j + 250) >= len(ecg_clean)):
                    continue
                temp_ecg = preprocessing.minmax_scale(ecg_clean[(j - 250):(j + 250)])
                index = np.where(rhythm_val == j)[0]
                X_temp.append(temp_ecg)
                y_temp.append([sample_id, beat_label[i], rhythm_label[index][0]])
        self.X = np.stack(X_temp, axis=0)
        self.Y = np.stack(y_temp, axis=0)

        if save:
            self.saveData(self.name + '_' + str(sample_id), save_path)

        return self.X, self.Y
