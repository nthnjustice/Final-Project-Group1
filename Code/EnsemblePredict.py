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
# model = load_model("/home/ubuntu/Deep-Learning/Final-Project-Group1/Code/dv_model_adam.hdf5")
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




#### disregard code below this line, does not work#####

# # ensemble used from https://machinelearningmastery.com/horizontal-voting-ensemble/
# def ensemble(models, test):
#     pred = [model.test_generator(test) for model in models]
#     pred = np.array(pred)
#     summed = np.sum(pred, axis=0)
#     result = np.argmax(summed, axis=1)
#     return result
#
# def evaluate_n_members(members, n_members, testX, testy):
#
#     subset = members[:n_members]
#     yhat = ensemble(subset, testX)
#     return accuracy_score(testy, yhat)
#
# # evaluate different numbers of ensembles on hold out set
# single_scores, ensemble_scores = list(), list()
# for i in range(1, len(models_list)+1):
#     # evaluate model with i members
#     ensemble_score = evaluate_n_members(models_list, i, test_generator, test_generator.classes)
#     # evaluate the i'th model standalone
#     testy_enc = to_categorical(test_generator.classes)
#     _, single_score = models_list[i-1].evaluate_generator(test_generator, testy_enc, verbose=0)
#     # summarize this step
#     print('> %d: single=%.3f, ensemble=%.3f' % (i, single_score, ensemble_score))
#     ensemble_scores.append(ensemble_score)
#     single_scores.append(single_score)
#
# # summarize average accuracy of a single final model
# print('Accuracy %.3f (%.3f)' % (np.mean(single_scores), np.std(single_scores)))
#
# # plot score vs number of ensemble members
# x_axis = [i for i in range(1, len(models_list)+1)]
# plt.plot(x_axis, single_scores, marker='o', linestyle='None')
# plt.plot(x_axis, ensemble_scores, marker='o')
# plt.show()



