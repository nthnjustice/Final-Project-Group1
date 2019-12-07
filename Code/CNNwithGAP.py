from keras.models import Sequential
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Convolution2D, MaxPooling2D, GlobalAveragePooling2D, BatchNormalization
from keras.utils import plot_model
from keras.callbacks import ModelCheckpoint
# from IPython.display import SVG
from keras.utils import model_to_dot
import matplotlib.pyplot as plt
from keras import optimizers
from imblearn.keras import balanced_batch_generator
from sklearn.utils import class_weight
import numpy as np


path_dir_train = '/home/ubuntu/Deep-Learning/Final-Project-Group1/Code/data/train'
path_dir_validate = '/home/ubuntu/Deep-Learning/Final-Project-Group1/Code/data/validation'
path_dir_test = '/home/ubuntu/Deep-Learning/Final-Project-Group1/Code/data/test'

img_width = 250
img_height = 250
epochs = 300
learning_rate = 0.01
decay = learning_rate/epochs

generator = ImageDataGenerator(rescale=1./255)

train_generator = generator.flow_from_directory(
    path_dir_train,
    target_size=(img_width, img_height),
    batch_size=64,
    color_mode="grayscale",
    class_mode='categorical')


validation_generator = generator.flow_from_directory(
    path_dir_validate,
    target_size=(img_width, img_height),
    batch_size=64,
    color_mode="grayscale",
    class_mode='categorical')


base = Sequential()
base.add(Convolution2D(32, (5, 5), input_shape=(img_width, img_height, 1)))
# base.add(BatchNormalization())
base.add(Activation('relu'))
base.add(MaxPooling2D(pool_size=(3, 3), strides=3))
base.add(Convolution2D(64, (5, 5)))
# base.add(BatchNormalization())
base.add(Activation('relu'))
# base.add(MaxPooling2D(pool_size=(2, 2)))
# base.add(Convolution2D(64, (3, 3)))
# base.add(BatchNormalization())
# base.add(Activation('relu'))
# base.add(MaxPooling2D(pool_size=(2, 2)))

GAP = Sequential()
GAP.add(GlobalAveragePooling2D())
GAP.add(Flatten())
GAP.add(Dense(32, activation='relu'))
GAP.add(Dense(4, activation='softmax'))

GAP_model = Sequential([
    base,
    GAP
])


SGD_decay = optimizers.SGD(lr=learning_rate, decay=decay, momentum=0.8)
AdamOP = optimizers.adam(lr=0.0001)
GAP_model.summary()
GAP_model.compile(optimizer=AdamOP,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

history = GAP_model.fit_generator(
    train_generator,
    # class_weight=class_weights,
    nb_epoch=epochs,
    validation_data=validation_generator,
    callbacks=[ModelCheckpoint("/home/ubuntu/Deep-Learning/Final-Project-Group1/models/taxa_area_GAP.hdf5",
                               monitor="val_loss", save_best_only=True)]
)

# Plot training & validation accuracy values
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.savefig('acc_taxa_area_oversampled_prelim.png')
plt.show()

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')

plt.savefig('val_taxa_area_oversampled_prelim.png')
plt.show()