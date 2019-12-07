from keras.models import Sequential
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Convolution2D, MaxPooling2D, GlobalAveragePooling2D
from keras.utils import plot_model
from keras.callbacks import ModelCheckpoint
# from IPython.display import SVG
from keras.utils import model_to_dot
import matplotlib.pyplot as plt
from keras import optimizers
from imblearn.keras import balanced_batch_generator
from sklearn.utils import class_weight
import numpy as np
import glob

base_dir = '/home/ubuntu/Deep-Learning/Final-Project-Group1/'
path_dir_models = base_dir + 'models'
path_dir_test = base_dir + 'Code/data/test'

models_list = glob.glob(path_dir_models + '/*')

generator = ImageDataGenerator(rescale=1./255)
test_generator = generator.flow_from_directory(
    directory=path_dir_test,
    target_size=(250, 250),
    color_mode="grayscale",
    batch_size=1,
    class_mode="categorical",
    shuffle=False,
    seed=42
)



STEP_SIZE_TEST=test_generator.n//test_generator.batch_size
test_generator.reset()
pred=model.predict_generator(test_generator, steps=STEP_SIZE_TEST, verbose=1)