
# coding: utf-8

# In[1]:

import time,os
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

import sys
np.random.seed(2017) 
from keras.models import Sequential
from keras.layers.convolutional import Convolution2D, MaxPooling2D, Conv2D
from keras.layers import Activation, Flatten, Dense, Dropout
from keras.layers.normalization import BatchNormalization
from keras.utils import np_utils
from keras.utils.np_utils import to_categorical
from deepModels import getModel


# In[2]:

#input_shape = (64, 64, 3)
#
#
#num_classes=2
#model = Sequential()
#model.add(Conv2D(48, (3, 3), padding='same', input_shape=input_shape))
#print('conv1: ', model.output_shape)
#model.add(Activation('relu'))
#model.add(Conv2D(48, (3, 3), padding='same'))
#print('conv2: ', model.output_shape)
#model.add(Activation('relu'))
#model.add(MaxPooling2D(pool_size=(2, 2)))
#print('pool1: ', model.output_shape)
#model.add(Dropout(0.25))
#model.add(Conv2D(96, (3, 3), padding='same'))
#print('conv3: ', model.output_shape)
#model.add(Activation('relu'))
#model.add(Conv2D(96, (3, 3), padding='same'))
#print('conv4: ', model.output_shape)
#model.add(Activation('relu'))
#model.add(MaxPooling2D(pool_size=(2, 2)))
#print('pool2: ', model.output_shape)
#model.add(Dropout(0.25))
#model.add(Conv2D(192, (3, 3), padding='same'))
#print('conv5: ', model.output_shape)
#model.add(Activation('relu'))
#model.add(Conv2D(192, (3, 3), padding='same'))
#model.add(Activation('relu'))
#model.add(MaxPooling2D(pool_size=(2, 2)))
#print('pool3: ', model.output_shape[1])
#model.add(Dropout(0.25))
#model.add(Conv2D(512, (8, 8), activation='relu', name='fc1'))
#print('dense 1: ', model.output_shape)
#model.add(Dropout(0.5))
#model.add(Conv2D(256, (1, 1), activation='relu', padding='same', name='fc2'))
#model.add(Dropout(0.5))
#model.add(Conv2D(num_classes, (1, 1), activation='softmax', name='predictions'))
##print('pre flatten: ', model.output_shape)
### remove this line for fcn
#model.add(Flatten())
##print('post flatten: ', model.output_shape)
##model.add(Dense(256))
#print('dense 2: ', model.output_shape)
##model.add(Activation('relu'))
##model.add(Dense(num_classes, activation='softmax'))
#print('final: ', model.output_shape)
## Compile the model
#
model = getModel()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

#sys.exit('bye')



# In[3]:

im_sz = 72
X = []
y = []
np.random.seed(42)
for filename in os.listdir(r'wildebeest_images/yes'):
    filename = "wildebeest_images/yes/" + filename
    img = Image.open(filename)
    img = img.resize([im_sz,im_sz])
    arr = np.array(img)
    X.append(arr)
    y.append(1)

for filename in os.listdir(r'wildebeest_images/no'):
    filename = "wildebeest_images/no/" + filename
    img = Image.open(filename)
    #img = img.resize([64, 64])
    img = img.resize([im_sz,im_sz])
    arr = np.array(img)
    X.append(arr)
    y.append(0)

for filename in os.listdir(r'wildebeest_images/no_contrast'):
    filename = "wildebeest_images/no_contrast/" + filename
    img = Image.open(filename)
    #img = img.resize([64, 64])
    img = img.resize([im_sz,im_sz])
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


from keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(zoom_range=0.2, vertical_flip=True,
                             horizontal_flip=True)


# train the model
start = time.time()
# Train the model
model_info = model.fit_generator(datagen.flow(X_train, y_train, batch_size = 128),
                                 steps_per_epoch = 512, nb_epoch = 200, 
                                 validation_data = (X_test, y_test), verbose=1)
end = time.time()
model.save_weights('cifar_weights.h5')


# In[6]:




# In[ ]:



