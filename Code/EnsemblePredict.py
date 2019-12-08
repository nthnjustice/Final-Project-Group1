from keras.models import Sequential, load_model
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
from sklearn.metrics import confusion_matrix, classification_report

base_dir = '/home/ubuntu/Deep-Learning/Final-Project-Group1/'
path_dir_models = base_dir + 'models'
path_dir_test = base_dir + 'Code/data/test'

models_list = glob.glob(path_dir_models + '/*')

generator = ImageDataGenerator()
test_generator = generator.flow_from_directory(
    directory=path_dir_test,
    target_size=(100, 100),
    color_mode="grayscale",
    batch_size=35,
    class_mode="categorical",
    shuffle=False,
    seed=42
)

model = load_model(path_dir_models + '/nj_model.hdf5')

steps = test_generator.n//test_generator.batch_size
test_generator.reset()
# pred = model.predict_generator(test_generator, steps=STEP_SIZE_TEST, verbose=1)
loss, acc = model.evaluate_generator(test_generator, steps=steps, verbose=0)
print("loss: ", loss)
print("acc: ", acc)




Y_pred = model.predict_generator(test_generator, steps+1)
y_pred = np.argmax(Y_pred, axis=1)
print('Confusion Matrix')
print(confusion_matrix(test_generator.classes, y_pred))
print('Classification Report')
print(classification_report(test_generator.classes, y_pred))
