from init_directories import init_dir
import os
import numpy as np
from sklearn.model_selection import train_test_split
import shutil


def run():
    root = 'data/images/'
    dirs = os.listdir(root)
    others = [name for name in dirs if len(os.listdir('data/images/' + name)) < 500]

    x = []
    y = []

    for i in dirs:
        images = os.listdir(root + i)
        label = 'OTHR' if i in others else i

        for image in images:
            x.append(image)
            y.append(label)

    xtrain, xtest, ytrain, ytest = train_test_split(np.array(x), np.array(y), test_size=0.2, random_state=0)
    xtrain, xvalid, ytrain, yvalid = train_test_split(xtrain, ytrain, test_size=0.2, random_state=0)

    dirs = [i for i in dirs if i not in others]
    dirs.append('OTHR')

    move_images('train/', dirs, xtrain, ytrain)
    move_images('test/', dirs)
    move_images('valid/', dirs)


def move_images(path, dirs, x, y):
    [init_dir(path + i) for i in dirs]

    for i in range(len(x)):
        source = 'data/images/' + x[i].split('_')[0] + '/' + x[i]
        destination = 'data/' + path + y[i] + '/' + x[i]
        shutil.copyfile(source, destination)
