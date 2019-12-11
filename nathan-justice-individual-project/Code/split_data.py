# helper that splits image data into train/validation/test splits

import os
import numpy as np
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler

from utils.move_files import move_files


def split_data(min_obsv, valid_size, test_size, oversample):
    data_path = 'data/images'
    dirs = os.listdir(data_path)
    others = [d for d in dirs if len(os.listdir(data_path + '/' + d)) < min_obsv]

    x = []
    y = []

    for i in dirs:
        images = os.listdir(data_path + '/' + i)
        label = 'OTHR' if i in others else i

        for image in images:
            x.append(image)
            y.append(label)

    x = np.array(x)
    y = np.array(y)

    xtrain, xvalid, ytrain, yvalid = train_test_split(x, y, test_size=valid_size, random_state=0)
    xtrain, xtest, ytrain, ytest = train_test_split(xtrain, ytrain, test_size=test_size, random_state=0)

    if oversample:
        ros = RandomOverSampler(random_state=0)
        xtrain = xtrain.reshape(-1, 1)
        xtrain, ytrain = ros.fit_resample(xtrain, ytrain)
        xtrain = xtrain.reshape(-1)

    dirs = [i for i in dirs if i not in others]
    if len(others) > 0:
        dirs.append('OTHR')

    in_root = 'data/images'
    move_files(in_root, 'data/train', dirs, xtrain, ytrain)
    move_files(in_root, 'data/validation', dirs, xvalid, yvalid)
    move_files(in_root, 'data/test', dirs, xtest, ytest)
