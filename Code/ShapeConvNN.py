from keras.models import Sequential
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import plot_model
from keras.callbacks import ModelCheckpoint
# from IPython.display import SVG
from keras.utils import model_to_dot
import matplotlib.pyplot as plt
from keras import optimizers


path_dir_train = '/home/ubuntu/Deep-Learning/Final-Project-Group1/Code/data/train'
path_dir_validate = '/home/ubuntu/Deep-Learning/Final-Project-Group1/Code/data/valid'
path_dir_test = '/home/ubuntu/Deep-Learning/Final-Project-Group1/Code/data/test'

img_width = 28
img_height = 28
epochs = 60

generator = ImageDataGenerator(rescale=1./255)

train_generator = generator.flow_from_directory(
    path_dir_train,
    target_size=(img_width, img_height),
    batch_size=100,
    color_mode="grayscale",
    interpolation="lanczos",
    class_mode='categorical')


validation_generator = generator.flow_from_directory(
    path_dir_validate,
    target_size=(img_width, img_height),
    batch_size=100,
    color_mode="grayscale",
    interpolation="lanczos",
    class_mode='categorical')

# #define model
model = Sequential()
model.add(Convolution2D(16, 3, 3, input_shape=(img_width, img_height, 1)))
model.add(Activation('relu'))
# model.add(Dropout(0.2))
model.add(MaxPooling2D(pool_size=(10, 10)))

model.add(Convolution2D(32, 2, 2))
model.add(Activation('relu'))
# model.add(Dropout(0.2))
# model.add(MaxPooling2D(pool_size=(2, 2)))

# model.add(Convolution2D(32, 2, 2))
# model.add(Activation('relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))


# model.add(Convolution2D(32, 2, 2))
# model.add(Activation('relu'))
# model.add(Dropout(0.2))
# model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
# model.add(Dense(56))
model.add(Dense(56))
model.add(Activation('relu'))
# model.add(Dropout(0.1))

model.add(Dense(8))
model.add(Activation('softmax'))

AdamOP=optimizers.adam(lr=0.001)
model.summary()
model.compile(optimizer=AdamOP,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit_generator(
    train_generator,
    nb_epoch=epochs,
    validation_data=validation_generator,
    callbacks=[ModelCheckpoint("/home/ubuntu/Deep-Learning/Final-Project-Group1/models/3_conv_layers_2x2conv.hdf5",
                               monitor="val_loss", save_best_only=True)]
)

# Plot training & validation accuracy values
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.savefig('basic_imgAG_acc_vs_val_acc.png')
plt.show()

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')

plt.savefig('basic_imgAG_loss_vs_val_loss.png')
plt.show()