from keras.models import Sequential
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Convolution2D, MaxPooling2D, AveragePooling2D, BatchNormalization, SpatialDropout2D
from keras.utils import plot_model
from keras.callbacks import ModelCheckpoint
# from IPython.display import SVG
from keras.utils import model_to_dot
import matplotlib.pyplot as plt
from keras import optimizers
from imblearn.keras import balanced_batch_generator
from sklearn.utils import class_weight
import numpy as np


# path_dir_train = '/home/ubuntu/Deep-Learning/Final-Project-Group1/Code/data/train'
# path_dir_validate = '/home/ubuntu/Deep-Learning/Final-Project-Group1/Code/data/validation'
# path_dir_test = '/home/ubuntu/Deep-Learning/Final-Project-Group1/Code/data/test'

path_dir_train = '/home/ubuntu/Deep-Learning/Final-Project-Group1/Code/data/validation'
path_dir_validate = '/home/ubuntu/Deep-Learning/Final-Project-Group1/Code/data/test'
# path_dir_test = '/home/ubuntu/Deep-Learning/Final-Project-Group1/Code/data/test'

img_width = 100
img_height = 100
epochs = 500
learning_rate = 0.01
decay = 1e-6
batch_size = 128
train_img_gen = ImageDataGenerator(horizontal_flip=True)
val_img_generator = ImageDataGenerator(horizontal_flip=False)
# rescale=1./255

train_generator = train_img_gen.flow_from_directory(
    path_dir_train,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    color_mode="grayscale",
    class_mode='categorical')


validation_generator = val_img_generator.flow_from_directory(
    path_dir_validate,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    color_mode="grayscale",
    class_mode='categorical')


base = Sequential()
base.add(Convolution2D(32, (10, 10), input_shape=(img_width, img_height, 1)))
base.add(Activation('relu'))
base.add(MaxPooling2D(pool_size=(5, 5)))

base.add(Convolution2D(64, (5, 5)))
base.add(Activation('relu'))
base.add(MaxPooling2D(pool_size=(2, 2)))

base.add(Convolution2D(128, (3, 3)))
base.add(Activation('relu'))
base.add(MaxPooling2D(pool_size=(2, 2)))

GAP = Sequential()
GAP.add(AveragePooling2D())
GAP.add(Flatten())
GAP.add(Dense(32, activation='relu'))
GAP.add(Dense(9, activation='softmax'))

GAP_model = Sequential([
    base,
    GAP
])


SGD_decay = optimizers.SGD(lr=0.01, decay=decay, momentum=0.9)
AdamOP = optimizers.adam(lr=0.001)
GAP_model.summary()
GAP_model.compile(optimizer=SGD_decay,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

history = GAP_model.fit_generator(
    train_generator,
    # class_weight=class_weights,
    nb_epoch=epochs,
    validation_data=validation_generator,
    callbacks=[ModelCheckpoint("/home/ubuntu/Deep-Learning/Final-Project-Group1/models/taxa_area_GAP_SGD.hdf5",
                               monitor="val_loss", save_best_only=True)]
)

# Plot training & validation accuracy values
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.savefig('acc_taxa_area_GAP_SGD.png')
plt.show()

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')

plt.savefig('val_taxa_area_GAP_SGD.png')
plt.show()