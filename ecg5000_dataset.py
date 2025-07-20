# @title ecg5000
!wget https://www.timeseriesclassification.com/aeon-toolkit/ECG5000.zip
!mkdir ecg
!unzip ECG5000.zip -d ecg

import numpy as np
path = '/content/ecg/ECG5000_TRAIN.txt'
def np_reader(path):
    data = np.loadtxt(path)
    y, x = data[:, 0], data[:, 1:]
    return x, y.astype(int)

xtrain, ytrain = np_reader('/content/ecg/ECG5000_TRAIN.txt')
xtest, ytest = np_reader('/content/ecg/ECG5000_TEST.txt')
# print(x.shape, y.shape)
