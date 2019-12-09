from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Convolution2D, Activation, SpatialDropout2D, MaxPooling2D, Flatten, Dense
from keras.layers import AveragePooling2D, Dropout, BatchNormalization
from keras.optimizers import SGD, Adam
from keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
import math

path_train = '/home/ubuntu/Deep-Learning/Final-Project-Group1/Code/data/train'
path_validation = '/home/ubuntu/Deep-Learning/Final-Project-Group1/Code/data/validation'
path_test = '/home/ubuntu/Deep-Learning/Final-Project-Group1/Code/data/test'
path_output = '/home/ubuntu/Deep-Learning/Final-Project-Group1/Code/'

img_width = 100
img_height = 100
target_size = (img_width, img_height)

epochs = 150
batch_size = 224
learning_rate = 0.1
decay = 1e-6
AdamOP = Adam(lr=0.0005)
SGD_decay = SGD(lr=learning_rate, decay=decay, momentum=0.9)

generator = ImageDataGenerator(horizontal_flip=True, vertical_flip=True)
train_generator = generator.flow_from_directory(
    path_train,
    target_size=target_size,
    batch_size=batch_size,
    color_mode='grayscale',
    shuffle=True,
    class_mode='categorical'
)

generator = ImageDataGenerator()
validation_generator = generator.flow_from_directory(
    path_validation,
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
    Convolution2D(16, kernel_size=(5, 5), input_shape=(img_width, img_height, 1)),
    BatchNormalization(),
    Activation('relu'),
    MaxPooling2D(pool_size=(5, 5)),
    SpatialDropout2D(0.2),

    Convolution2D(32, kernel_size=(5, 5)),
    BatchNormalization(),
    Activation('relu'),
    # MaxPooling2D(pool_size=(2, 2)),
    # SpatialDropout2D(0.2),

    # Convolution2D(128, kernel_size=(3, 3)),
    # BatchNormalization(),
    # Activation('relu'),
    AveragePooling2D(pool_size=(5, 5)),
    SpatialDropout2D(0.2),

    Flatten(),
    Dense(700),
    Activation('relu'),
    Dropout(0.5),
    Dense(8),
    Activation('softmax')
])

model.compile(optimizer=AdamOP, loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()
history = model.fit_generator(
    train_generator,
    epochs=epochs,
    validation_data=validation_generator,
    callbacks=[ModelCheckpoint(path_output + 'dv_model_adam.hdf5', monitor="val_loss", save_best_only=True)]
)

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.savefig(path_output + 'dv_acc_adam.png')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.savefig(path_output + 'dv_loss_adam.png')
plt.show()
