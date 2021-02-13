from glob import glob

import numpy as np
from keras.applications.resnet50 import preprocess_input
from keras.callbacks import ModelCheckpoint
from keras.layers import Dense
from keras.layers import GlobalAveragePooling2D
from keras.models import Sequential
from keras.preprocessing import image
from keras.utils import np_utils
from sklearn.datasets import load_files
from tqdm import tqdm
import cv2

from extract_bottleneck_features import *


def Dog_Recognize_app(imgpath):
    predict = Resnet50_prediction_breed(imgpath)
    img = cv2.imread(imgpath)
    img = cv2.resize(img, (450,450))
    cv2.putText(img, 'Predizione razza: {}'.format(dog_names[predict]), (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
    #cv2.putText(img, 'Razza: {}'.format(str(label_test)), (20, 60),
    #cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 116, 55), 3)

    cv2.imshow("Predizione ",img)
    print("Questo cane sembra proprio appartenente alla razza: ")
    print(dog_names[predict])
    cv2.waitKey(0)



def Resnet50_prediction_breed(img_path):
    # extract bottleneck features
    bottleneck_feature = extract_Resnet50(path_to_tensor(img_path))
    # obtain predicted vector
    predicted_vector = Resnet50_model.predict(bottleneck_feature)
    # return dog breed that is predicted by the model
    return np.argmax(predicted_vector,-1)


def get_bottleneck_features(path):
    bottleneck_features = np.load(path)
    train = bottleneck_features['train']
    valid = bottleneck_features['valid']
    test = bottleneck_features['test']
    return train, valid, test


def path_to_tensor(img_path):
    # loads RGB image as PIL.Image.Image type
    img = image.load_img(img_path, target_size=(224, 224))
    # convert PIL.Image.Image type to 3D tensor with shape (224, 224, 3)
    x = image.img_to_array(img)
    # convert 3D tensor to 4D tensor with shape (1, 224, 224, 3) and return 4D tensor
    return np.expand_dims(x, axis=0)


def paths_to_tensor(img_paths):
    list_of_tensors = [path_to_tensor(img_path) for img_path in tqdm(img_paths)]
    return np.vstack(list_of_tensors)


def test_model(model, test_tensors, test_targets, name):
    # get index of predicted dog breed for each image in test set
    predictions = [np.argmax(model.predict(np.expand_dims(tensor, axis=0))) for tensor in test_tensors]

    # report test accuracy
    test_accuracy = 100*np.sum(np.array(predictions)==np.argmax(test_targets, axis=1))/len(predictions)
    print(f'Test accuracy {name}: {round(test_accuracy, 4)}%')


# define function to load train, test, and validation datasets
def load_dataset(path):
    data = load_files(path)
    dog_files = np.array(data['filenames'])
    dog_targets = np_utils.to_categorical(np.array(data['target']), 133)
    return dog_files, dog_targets


def ResNet50_predict_labels(img_path):
    # returns prediction vector for image located at img_path
    img = preprocess_input(path_to_tensor(img_path))
    return np.argmax(Resnet50_model.predict(img))


# define function to load train, test, and validation datasets
def load_dataset(path):
    data = load_files(path)
    dog_files = np.array(data['filenames'])
    dog_targets = np_utils.to_categorical(np.array(data['target']), 133)
    return dog_files, dog_targets


# load train, test, and validation datasets
train_files, train_targets = load_dataset('dogImages/train')
valid_files, valid_targets = load_dataset('dogImages/valid')
test_files, test_targets = load_dataset('dogImages/test')

# load list of dog names
dog_names = [item[20:-1] for item in sorted(glob("dogImages/train/*/"))]
dog_names = np.array(dog_names)

# print statistics about the dataset
print('There are %d total dog categories.' % len(dog_names))
print('There are %s total dog images.\n' % len(np.hstack([train_files, valid_files, test_files])))
print('There are %d training dog images.' % len(train_files))
print('There are %d validation dog images.' % len(valid_files))
print('There are %d test dog images.'% len(test_files))


train_Resnet50, valid_Resnet50, test_Resnet50 = get_bottleneck_features('DogResnet50Data.npz')

Resnet50_model = Sequential()
Resnet50_model.add(GlobalAveragePooling2D(input_shape=(train_Resnet50.shape[1:])))
Resnet50_model.add(Dense(133, activation='softmax'))
Resnet50_model.summary()


Resnet50_model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

checkpointer = ModelCheckpoint(filepath='weights.best.Resnet50.hdf5', verbose=1, save_best_only=True)
#Resnet50_model.fit(train_Resnet50, train_targets, validation_data=(valid_Resnet50, valid_targets),epochs=100, batch_size=20, callbacks=[checkpointer], verbose=1)
Resnet50_model.load_weights('weights.best.Resnet50.hdf5')
test_model(Resnet50_model,test_Resnet50, test_targets, 'Resnet50')

#Qui eseguo i test per riscontrate se effettivamente la macchina è accurata,controllando l output ricevuto dalla mia funziode di gestione Dog_Recognize_app
#Dog_Recognize_app('images/American_water_spaniel_00648.jpg')
#Dog_Recognize_app('images/Brittany_02625.jpg')
#Dog_Recognize_app('images/Curly-coated_retriever_03896.jpg')
Dog_Recognize_app('dogImages/valid/004.Akita/Akita_00247.jpg')
Dog_Recognize_app('dogImages/valid/039.Bull_terrier/Bull_terrier_02732.jpg')
Dog_Recognize_app('dogImages/valid/054.Collie/Collie_03791.jpg')
