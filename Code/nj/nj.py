import math
import numpy as np
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Convolution2D, Activation, SpatialDropout2D, MaxPooling2D, Flatten, Dense
from keras.layers import AveragePooling2D, Dropout, BatchNormalization, GlobalAveragePooling2D
from keras.optimizers import SGD, Adam
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
from sklearn.metrics import confusion_matrix, classification_report


path_train = '/home/ubuntu/Final-Project-Group1/Code/data/train'
path_validation = '/home/ubuntu/Final-Project-Group1/Code/data/validation'
path_test = '/home/ubuntu/Final-Project-Group1/Code/data/test'
path_output = '/home/ubuntu/Final-Project-Group1/Code/nj/'

img_width = 100
img_height = 100
target_size = (img_width, img_height)

epochs = 60
batch_size = 32

generator = ImageDataGenerator(horizontal_flip=True, vertical_flip=True)
train_generator = generator.flow_from_directory(
    path_train,
    target_size=target_size,
    batch_size=batch_size,
    color_mode='grayscale',
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

generator = ImageDataGenerator()
test_generator = generator.flow_from_directory(
    path_test,
    target_size=target_size,
    batch_size=batch_size,
    color_mode='grayscale',
    class_mode='categorical',
    shuffle=False,
    seed=42
)

model = Sequential([
    Convolution2D(32, kernel_size=(5, 5), input_shape=(img_width, img_height, 1)),
    BatchNormalization(),
    Activation('relu'),
    MaxPooling2D(pool_size=(5, 5)),
    SpatialDropout2D(0.2),

    Convolution2D(64, kernel_size=(5, 5)),
    BatchNormalization(),
    Activation('relu'),
    AveragePooling2D(pool_size=(5, 5)),
    SpatialDropout2D(0.2),

    Flatten(),
    Dense(700),
    Activation('relu'),
    Dropout(0.5),
    Dense(11),
    Activation('softmax')
])

model.compile(optimizer=Adam(lr=0.001), loss='categorical_crossentropy', metrics=['acc'])

history = model.fit_generator(
    train_generator,
    epochs=epochs,
    validation_data=validation_generator,
    callbacks=[ModelCheckpoint(path_output + 'nj_model.hdf5', monitor="val_loss")]
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

model = load_model(path_output + 'nj_model.hdf5')
steps = test_generator.n // test_generator.batch_size
test_generator.reset()
loss, acc = model.evaluate_generator(test_generator, steps=steps, verbose=0)
print("loss: ", loss)
print("acc: ", acc)

Y_pred = model.predict_generator(test_generator, steps + 1)
y_pred = np.argmax(Y_pred, axis=1)
print('Confusion Matrix')
print(confusion_matrix(test_generator.classes, y_pred))
print('Classification Report')
print(classification_report(test_generator.classes, y_pred))
