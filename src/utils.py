def create_directory(dirName):
    import os
    try:
        os.mkdir(dirName)
        print("Directory " , dirName ,  " Created ")
    except FileExistsError:
        print("Directory " , dirName ,  " already exists")

def load(name, path='../temp/'):
    import numpy as np
    '''
    Loads data and associated labelling from hard disc
    '''
    X = np.load(path + 'x_' + name + '.npy', allow_pickle=True)
    y = np.load(path + 'y_' + name + '.npy', allow_pickle=True)
    return X, y

def save(X, y, name, path='../temp/'):
    import numpy as np
    '''
    Writes data and associated labelling to hard disc
    '''
    np.save(path + 'x_' + name + '.npy', X)
    np.save(path + 'y_' + name + '.npy', y)

    return True
