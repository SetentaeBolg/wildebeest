
# coding: utf-8

# In[1]:

import time,os
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

np.random.seed(2017)
from keras.models import Sequential
from keras.layers.convolutional import Convolution2D, MaxPooling2D, Conv2D
from keras.layers import Activation, Flatten, Dense, Dropout
from keras.layers.normalization import BatchNormalization
from keras.utils import np_utils
from keras.utils.np_utils import to_categorical


# In[2]:



input_shape = (64, 64, 3)

fcnmodel = Sequential()
fcnmodel.add(Conv2D(48, (3, 3), padding='same', activation='relu', input_shape=input_shape))
fcnmodel.add(Conv2D(48, (3, 3), activation='relu', padding='same'))
fcnmodel.add(MaxPooling2D(pool_size=(2, 2)))
fcnmodel.add(Dropout(0.25))
fcnmodel.add(Conv2D(96, (3, 3), activation='relu', padding='same'))
fcnmodel.add(Conv2D(96, (3, 3), activation='relu', padding='same'))
fcnmodel.add(MaxPooling2D(pool_size=(2, 2)))
fcnmodel.add(Dropout(0.25))
fcnmodel.add(Conv2D(192, (3, 3), activation='relu', padding='same'))
fcnmodel.add(Conv2D(192, (3, 3), activation='relu', padding='same'))
fcnmodel.add(MaxPooling2D(pool_size=(2, 2)))
fcnmodel.add(Dropout(0.25))
fcnmodel.add(Conv2D(512, (6, 6), activation='relu', name='fc1'))
fcnmodel.add(Dropout(0.5))
fcnmodel.add(Conv2D(256, (1, 1), activation='relu', padding='same', name='fc2'))
fcnmodel.add(Dropout(0.5))
fcnmodel.add(Conv2D(2, (1, 1), activation='softmax', name='predictions'))
fcnmodel.add(Flatten())



num_classes=2
model = Sequential()
model.add(Convolution2D(48, 3, 3, border_mode='same', input_shape=input_shape))
model.add(Activation('relu'))
model.add(Convolution2D(48, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Convolution2D(96, 3, 3, border_mode='same'))
model.add(Activation('relu'))
model.add(Convolution2D(96, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Convolution2D(192, 3, 3, border_mode='same'))
model.add(Activation('relu'))
model.add(Convolution2D(192, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(512, name='fc1'))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(256, name='fc2'))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax', name='predictions'))
# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


# In[3]:

X = []
y = []
#np.random.seed(42)
for filename in os.listdir(r'wildebeest_images/yes'):
    filename = "wildebeest_images/yes/" + filename
    img = Image.open(filename)
    img = img.resize([64,64])
    arr = np.array(img)
    X.append(arr)
    y.append(1)

for filename in os.listdir(r'wildebeest_images/no'):
    filename = "wildebeest_images/no/" + filename
    img = Image.open(filename)
    img = img.resize([64, 64])
    arr = np.array(img)
    X.append(arr)
    y.append(0)

for filename in os.listdir(r'wildebeest_images/no_contrast'):
    filename = "wildebeest_images/no_contrast/" + filename
    img = Image.open(filename)
    img = img.resize([64, 64])
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

# In[8]:




end = time.time()
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
            weights[0] = np.reshape(weights[0],(6,6,192,512))
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

fcnmodel.save_weights('fcn_cifar10_weights.h5')

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


# In[6]:




# In[ ]:



