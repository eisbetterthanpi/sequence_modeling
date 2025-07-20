# @title eigenworm
# https://www.timeseriesclassification.com/dataset.php
!wget https://timeseriesclassification.com/aeon-toolkit/Worms.zip
!mkdir worms
!unzip Worms.zip -d worms

# 131 train and 128 test. We have truncated each series to the shortest usable. Each series has 17984 observations. Each worm is classified as either wild-type (the N2 reference strain) or one of four mutant types: goa-1; unc-1; unc-38 and unc-63.
import numpy as np
# https://github.com/huckiyang/Voice2Series-Reprogramming/blob/main/ts_dataloader.py#L28
path = '/content/worms/Worms_TRAIN.txt'
def np_reader(path):
    data = np.loadtxt(path)
    y, x = data[:, 0], data[:, 1:]
    return x, y.astype(int)

xtrain, ytrain = np_reader('/content/worms/Worms_TRAIN.txt')
xtest, ytest = np_reader('/content/worms/Worms_TEST.txt')
