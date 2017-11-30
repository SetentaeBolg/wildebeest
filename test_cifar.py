
# coding: utf-8

# In[1]:

import time,os
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

np.random.seed(2017) 
from keras.models import Sequential
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers import Activation, Flatten, Dense, Dropout
from keras.layers.normalization import BatchNormalization
from keras.utils import np_utils
from keras.utils.np_utils import to_categorical


# In[2]:

input_shape = (64, 64, 3)


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
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(256))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))
# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


# In[3]:

X = []
y = []
np.random.seed(42)
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
result = model.predict(X_test)
predicted_class = np.argmax(result, axis=1)
true_class = np.argmax(y_test, axis=1)
num_correct = np.sum(predicted_class == true_class) 
accuracy = float(num_correct)/result.shape[0]
print(accuracy * 100)


# In[6]:




# In[ ]:



