import time,os,sys
from PIL import Image
import numpy as np
import cv2
from keras.utils.np_utils import to_categorical

np.random.seed(2017)
from deepModels import getModel, getSegModel

nx = 512
ny = 512

fcnmodel = getSegModel(ny,nx)


num_classes=2
# Compile the model

model = getModel()
model.compile(optimizer='nadam', loss='categorical_crossentropy', metrics=['accuracy'])

model.load_weights('cifar_weights.h5')

flattened_layers = fcnmodel.layers
index = {}
for layer in flattened_layers:
    if layer.name:
        index[layer.name] = layer
for layer in model.layers:
    weights = layer.get_weights()
    for i in range(len(weights)):
        print(layer.name, ' ', i, ': ', weights[i].shape)
    if layer.name in ['fc1','fc2','predictions']:
        if layer.name == 'fc1':
            weights[0] = np.reshape(weights[0],(9,9,192,512))
        elif layer.name == 'fc2':
            weights[0] = np.reshape(weights[0],(1,1,512,256))
        else:
            weights[0] = np.reshape(weights[0],(1,1,256,2))
    if layer.name in index:
        index[layer.name].set_weights(weights)

for layer in fcnmodel.layers:
    weights = layer.get_weights()
    for i in range(len(weights)):
        print(layer.name, ' ', i, ': ', weights[i].shape)

fcnmodel.save_weights('fcn_cifar10_weights_from_classifier.h5')

X = []
y = []
#np.random.seed(42)
for filename in os.listdir(r'wildebeest_images/yes'):
    filename = "wildebeest_images/yes/" + filename
    img = Image.open(filename)
    img = img.resize([72,72])
    arr = np.array(img)
    X.append(arr)
    y.append(1)
#
for filename in os.listdir(r'wildebeest_images/no'):
    filename = "wildebeest_images/no/" + filename
    img = Image.open(filename)
    img = img.resize([72,72])
    arr = np.array(img)
    X.append(arr)
    y.append(0)

for filename in os.listdir(r'wildebeest_images/no_contrast'):
    filename = "wildebeest_images/no_contrast/" + filename
    img = Image.open(filename)
    img = img.resize([72,72])
    arr = np.array(img)
    X.append(arr)
    y.append(0)

X = np.asarray(X)
y = np.asarray(y)
X = X.astype('float32')/255

shuffle_index = np.random.permutation(X.shape[0])
X = X[shuffle_index]
y = y[shuffle_index]
y = to_categorical(y,2)
(X_train, y_train), (X_test, y_test) = (X[:8000], y[:8000]), (X[8000:], y[8000:])
print('Train set size : ', X_train.shape[0])
print('Test set size : ', X_test.shape[0])






result = model.predict(X_test)
predicted_class = np.argmax(result, axis=1)
true_class = np.argmax(y_test, axis=1)
print(true_class.shape)
num_correct = np.sum(predicted_class == true_class)
print(num_correct)
print(result.shape[0])
accuracy = float(num_correct)/result.shape[0]

tpos = 0
tneg = 0
fpos = 0
fneg = 0
for i in range (y_test.shape[0]):
    if y_test[i, 1] == 1:
        if result[i, 1] > result[i, 0]:
            tpos = tpos + 1
        else:
            fneg = fneg + 1
    else:
        if result[i, 1] > result[i, 0]:
            fpos = fpos + 1
        else:
            tneg = tneg + 1


print('Number of correct identified wildebeests: ', tpos)
print('Number of false positives: ', fpos)
print('Number of false negatives: ', fneg)
print('Number of true negatives: ', tneg)
print(accuracy * 100)



