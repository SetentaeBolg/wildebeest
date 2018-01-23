import time,os,sys
from PIL import Image
import numpy as np
import cv2
from keras.utils.np_utils import to_categorical
from keras.applications.vgg19 import VGG19,preprocess_input, decode_predictions
from keras.applications.vgg16 import VGG16
from keras.layers import Flatten, Dense, Input, Dropout, Conv2D, MaxPooling2D
from keras.models import Sequential, Model
from keras import applications
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping
from keras.optimizers import SGD
import random

np.random.seed(2017)
from deepModels import getModel, getSegModel

nx = 512
ny = 512



num_classes=2
# Compile the model

model_list = []

model = getModel()
model.compile(optimizer='nadam', loss='binary_crossentropy', metrics=['accuracy'])

#model.load_weights('cifar_weights.h5')

model_list.append(model)

base_model = VGG16(weights = 'imagenet', include_top = False, input_shape=(64,64,3))
model2 = Sequential()
for layer in base_model.layers:
    model2.add(layer)

for layer in model2.layers[:10]:
    layer.trainable = False

model2.add(Flatten())
model2.add(Dense(4096, activation = 'relu', name = 'fc1'))
model2.add(Dense(4096, activation = 'relu', name = 'fc2'))
model2.add(Dense(2, activation = 'softmax', name = 'predictions'))

model_list.append(model2)

# set-up the model
fcn_model = Sequential()
for l in base_model.layers:
    fcn_model.add(l)

for layer in fcn_model.layers:
    layer.trainable = False

fcn_model.add(Conv2D(256, (2,2), activation='relu', name='fc1',input_shape=base_model.output_shape[1:]))
fcn_model.add(Dropout(0.5))
fcn_model.add(Conv2D(num_classes, (1, 1), activation='sigmoid', name='predictions'))
fcn_model.add(Flatten())

fcn_model.compile(optimizer='rmsprop',loss='binary_crossentropy',metrics=['accuracy'])

model_list.append(fcn_model)

#create training samples
train_images = []
train_labels = []

import glob

# use 10000 images
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

datagen = ImageDataGenerator(zoom_range=0.0, vertical_flip=False, horizontal_flip=False)
callbacks = [EarlyStopping(monitor='val_loss', patience=2, verbose=1)]
sgd = SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)
model_list[1].compile(optimizer = sgd, loss = 'binary_crossentropy', metrics=['accuracy'])
model_list[0].compile(optimizer = sgd, loss = 'binary_crossentropy', metrics=['accuracy'])

xup = model_list[1].inputs[0].shape[1].value 
yup = model_list[1].inputs[0].shape[2].value

model_list[2].summary()

model_names = ['CIFAR10 classifier, high contrast', 'VGG16 classifier, high contrast', 'VGG16 FCN, high contrast']

if not(os.path.isfile('cifar10_classifier_highcontrast.h5')):
    # train the model
    start = time.time()
    print('Cifar training starts')
    model_list[0].fit_generator(datagen.flow(X_train, y_train, batch_size = 1024),
                                     steps_per_epoch = 512, epochs = 200, callbacks=callbacks,
                                     validation_data = (X_test, y_test), verbose=1)
    end = time.time()
    print('Training ends')
    model_list[0].save_weights('cifar10_classifier_highcontrast.h5')
else:
    model_list[0].load_weights('cifar10_classifier_highcontrast.h5')

if not(os.path.isfile('vgg16_classifier_highcontrast.h5')):
    # train the model
    start = time.time()
    print('VGG16 classifier training starts')
    X_t = X[:,((72 - xup) // 2):((72 - xup) // 2) + xup,((72 - yup) // 2):((72 - yup) // 2) + yup,:]
    (X_train, y_train), (X_test, y_test) = (X_t[:8000], y[:8000]), (X_t[8000:], y[8000:])
    model_list[1].fit_generator(datagen.flow(X_train, y_train, batch_size = 1024),
                                     steps_per_epoch = 512, epochs = 200, callbacks=callbacks,
                                     validation_data = (X_test, y_test), verbose=1)
    end = time.time()
    print('Training ends')
    model_list[1].save_weights('vgg16_classifier_highcontrast.h5')
else:
    model_list[1].load_weights('vgg16_classifier_highcontrast.h5')

if not(os.path.isfile('vgg16_fcn_highcontrast.h5')):
    # train the model
    start = time.time()
    print('VGG16 FCN training starts')
    X_t = X[:,((72 - xup) // 2):((72 - xup) // 2) + xup,((72 - yup) // 2):((72 - yup) // 2) + yup,:]
    (X_train, y_train), (X_test, y_test) = (X_t[:8000], y[:8000]), (X_t[8000:], y[8000:])
    model_list[2].fit_generator(datagen.flow(X_train, y_train, batch_size = 1024),
                                     steps_per_epoch = 512, epochs = 200, callbacks=callbacks,
                                     validation_data = (X_test, y_test), verbose=1)
    end = time.time()
    print('Training ends')
    model_list[2].save_weights('vgg16_fcn_highcontrast.h5')
else:
    model_list[2].load_weights('vgg16_fcn_highcontrast.h5')

for model in model_list:
    X_t = X
    if model.inputs[0].shape[1].value < 72:
        xup = model.inputs[0].shape[1].value 
        yup = model.inputs[0].shape[2].value
        X_t = X_t[:,((72 - xup) // 2):((72 - xup) // 2) + xup,((72 - yup) // 2):((72 - yup) // 2) + yup,:]
    result = model.predict(X_t)
    predicted_class = np.argmax(result, axis=1)
    true_class = np.argmax(y, axis=1)
    num_correct = np.sum(predicted_class == true_class)
    accuracy = float(num_correct)/result.shape[0]
    tpos = 0
    tneg = 0
    fpos = 0
    fneg = 0
    for i in range (y.shape[0]):
        if y[i, 1] == 1:
            if result[i, 1] > result[i, 0]:
                tpos = tpos + 1
            else:
                fneg = fneg + 1
        else:
            if result[i, 1] > result[i, 0]:
                fpos = fpos + 1
            else:
                tneg = tneg + 1
    print('Model: ' + model_names[model_list.index(model)])
    print('Number of correct identified wildebeests: ', tpos)
    print('Number of false positives: ', fpos)
    print('Number of false negatives: ', fneg)
    print('Number of true negatives: ', tneg)
    print('Overall accuracy: ' + str(accuracy * 100) + '%')
    print()
