# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import scipy.io as spio
import os

# There are three versions of MNIST dataset
dataTypes = ['digits-normal.mat', 'digits-scaled.mat', 'digits-jitter.mat']

classNames = [0,1,2,3,4,5,6,7,8,9]

trainSet = 1
testSet = 2
validSet = 3

def todict(matobj):
    dict = {}
    for strg in matobj._fieldnames:
        elem = matobj.__dict__[strg]
        if isinstance(elem, spio.matlab.mio5_params.mat_struct):
            dict[strg] = todict(elem)
        else:
            dict[strg] = elem
    return dict

def loadmat(path):
    return todict(spio.loadmat(path,
                               struct_as_record=False,
                               squeeze_me=True)['data'])

#for i in range(len(dataTypes)):
dataType = dataTypes[2]
    #Load data
path = os.path.join('..', 'data', dataType)
data = loadmat(path)
print('+++ Loading digits of dataType: ',dataType)

train_images = []
train_labels = []
test_images = []
test_labels = []
valid_images = []
valid_labels = []

for j in range(data['x'].shape[2]):
    if data['set'][j] == trainSet:
        train_images.append([data['x'][:,:,j]])
        train_labels.append(data['y'][j])
    if data['set'][j] == testSet:
        test_images.append([data['x'][:,:,j]])
        test_labels.append(data['y'][j])
    if data['set'][j] == validSet:
        valid_images.append([data['x'][:,:,j]])
        valid_labels.append(data['y'][j])

train_images = np.asarray(train_images)
train_labels = np.asarray(train_labels)
test_images = np.asarray(test_images)
test_labels = np.asarray(test_labels)
valid_images = np.asarray(valid_images)
valid_labels = np.asarray(valid_labels)

model = keras.Sequential([
    keras.layers.Conv2D(filters=32, kernel_size=(3,3), padding='same',\
                        input_shape=(1, 28, 28),activation='relu'),
    keras.layers.Conv2D(filters=32, kernel_size=(3,3), padding='same',\
                        activation='relu'),
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(train_images, train_labels, epochs=10)

test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
valid_loss, valid_acc = model.evaluate(valid_images,  valid_labels, verbose=2)

print('\nTest accuracy:', test_acc)
print('\nValidation accuracy:', valid_acc)

