from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Convolution2D, Activation, SpatialDropout2D, MaxPooling2D, Flatten, Dense
from keras.layers import BatchNormalization, AveragePooling2D, Dropout
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
import math

path_train = '/home/ubuntu/Final-Project-Group1/Code/data/train'
path_validation = '/home/ubuntu/Final-Project-Group1/Code/data/validation'
path_test = '/home/ubuntu/Final-Project-Group1/Code/data/test'
path_output = '/home/ubuntu/Final-Project-Group1/Code/nj/'

img_width = 100
img_height = 100
target_size = (img_width, img_height)

epochs = 10
batch_size = 32

generator = ImageDataGenerator(horizontal_flip=True, vertical_flip=True)
train_generator = generator.flow_from_directory(
    path_validation,
    target_size=target_size,
    batch_size=batch_size,
    color_mode='grayscale',
    class_mode='categorical'
)

generator = ImageDataGenerator()
validation_generator = generator.flow_from_directory(
    path_test,
    target_size=target_size,
    batch_size=batch_size,
    color_mode='grayscale',
    class_mode='categorical'
)

# test_generator = generator.flow_from_directory(
#     path_test,
#     target_size=target_size,
#     batch_size=batch_size,
#     color_mode='grayscale',
#     class_mode='categorical'
# )

model = Sequential([
    Convolution2D(32, kernel_size=(15, 15), strides=1, input_shape=(img_width, img_height, 1)),
    Activation('relu'),
    #SpatialDropout2D(0.2),
    #BatchNormalization(),
    MaxPooling2D(pool_size=3),
    Convolution2D(64, kernel_size=(5, 5), strides=1),
    Activation('relu'),
    #SpatialDropout2D(0.2),
    #BatchNormalization(),
    AveragePooling2D(),
    Flatten(),
    Dense(32),
    Activation('relu'),
    #Dropout(0.5),
    Dense(9),
    Activation('softmax')
])

model.compile(optimizer=Adam(lr=0.01), loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit_generator(
    train_generator,
    epochs=epochs,
    #steps_per_epoch=math.ceil(validation_generator.n / batch_size),
    validation_data=validation_generator,
    #validation_steps=math.ceil(test_generator.n / batch_size),
    callbacks=[ModelCheckpoint(path_output + 'nj_model.hdf5', monitor="val_loss", save_best_only=True)]
)

plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.savefig(path_output + 'nj_acc.png')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.savefig(path_output + 'nj_loss.png')
plt.show()
