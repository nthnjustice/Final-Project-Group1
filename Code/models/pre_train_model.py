import os
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Convolution2D
from keras.applications import VGG16


def pre_train_model():
    print('running pre_train_model model')
    conv_base = VGG16(weights='imagenet',
                      include_top=False,
                      input_shape=(100, 100, 3))
    conv_base.summary()

    train_dir = 'data/train'
    validation_dir = 'data/validation'
    test_dir = "data/test"

    datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range = 0.2,
        horizontal_flip = True,
        fill_mode = 'nearest'
    )
    batch_size = 20

    def extract_features(directory, sample_count):
        features = np.zeros(shape=(sample_count, 3, 3, 512))
        labels = np.zeros(shape=(sample_count,9))
        generator = datagen.flow_from_directory(
            directory,
            target_size=(100, 100),
            batch_size=batch_size,
            color_mode="rgb",
            interpolation="lanczos",
            class_mode='categorical')
        i = 0
        for inputs_batch, labels_batch in generator:
            features_batch = conv_base.predict(inputs_batch)
            features[i * batch_size : (i + 1) * batch_size] = features_batch

            labels[i * batch_size : (i + 1) * batch_size] = labels_batch
            i += 1
            if i * batch_size >= sample_count:
                # Note that since generators yield data indefinitely in a loop,
                # we must `break` after every image has been seen once.
                break
        return features, labels

    train_features, train_labels = extract_features(train_dir,115524)
    validation_features, validation_labels = extract_features(validation_dir, 32091)
    test_features, test_labels = extract_features(test_dir, 12836)

    train_features = np.reshape(train_features, (115524, 3 * 3 * 512))
    validation_features = np.reshape(validation_features, (32091, 3 * 3 * 512))
    test_features = np.reshape(test_features, (12836, 3 * 3 * 512))

    # train_features, train_labels = extract_features(train_dir,1000)
    # validation_features, validation_labels = extract_features(validation_dir, 300)
    # test_features, test_labels = extract_features(test_dir,100)
    #
    # train_features = np.reshape(train_features, (1000, 3* 3 * 512))
    # validation_features = np.reshape(validation_features, (300, 3* 3 * 512))
    # test_features = np.reshape(test_features, (100, 3 * 3 * 512))

    from keras import models
    from keras import layers
    from keras import optimizers

    model = models.Sequential()
    #
    # model.add(layers.Dense(Convolution2D(256, (3, 3)))
    # model.add(layers.Activation('relu'))
    # model.add(layers.AveragePooling2D(pool_size=(2, 2)))
    # model.add(layers.Dropout(0.5))



    model.add(layers.Dense(256, activation='relu',input_dim=3*3*512))
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(9, activation='softmax'))

    model.compile(optimizer=optimizers.RMSprop(lr=2e-5),
                  loss='categorical_crossentropy',
                  metrics=['acc'])

    history = model.fit(train_features, train_labels,
                        epochs=30,
                        batch_size=20,
                        validation_data=(validation_features, validation_labels))


    import matplotlib.pyplot as plt

    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(len(acc))

    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()

    plt.figure()

    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()

    plt.show()

