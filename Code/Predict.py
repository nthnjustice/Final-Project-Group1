from keras.models import Sequential, load_model
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import glob
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

base_dir = '/home/ubuntu/Deep-Learning/Final-Project-Group1/'
path_dir_models = base_dir + 'models'
path_dir_test = base_dir + 'data/test'

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
loss, acc = model.evaluate_generator(test_generator, steps=steps, verbose=0)
print("accuracy: ", acc)
print("loss: ", loss)

pred = model.predict_generator(test_generator, steps+1)
prediction = np.argmax(pred, axis=1)
print('FINAL TESTING REPORT')
print(classification_report(test_generator.classes, pred))

