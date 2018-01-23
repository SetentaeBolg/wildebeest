import time,os,sys
from PIL import Image
import numpy as np
import cv2
import pandas as pd
from keras.utils.np_utils import to_categorical
from deepModels import *
from sklearn.utils import shuffle

nx = 512
ny = 512

fcnmodel = getVgg16SegModel(ny,nx)
num_classes=2

fcnmodel.load_weights('vgg16-header_classifier.h5')
not_wildebeest_locs = pd.read_csv('new_not_wildebeest_locations.csv')
not_wildebeest_locs = shuffle(not_wildebeest_locs)

print(fcn_model.summary())

fcn_model.compile(optimizer='rmsprop',loss='binary_crossentropy',metrics=['accuracy'])

#create training samples
train_images = []
train_labels = []

import glob

for filename in glob.iglob('2015/cls_train_images/w/*.jpg'):
    img = cv2.imread(filename)
    if img is not None:
        train_images.append(np.array(img))
        train_labels.append(1)

# use equal non-wildebeest images for training
not_wildebeest_locs = not_wildebeest_locs[:len(train_labels)]
not_wildebeest_images = not_wildebeest_locs['image_name'].drop_duplicates()
for i1, image_row in not_wildebeest_images.iterrows():
    not_wildebeest_sub_locs = not_wildebeest_locs[not_wildebeest_locs.image_name == image_row.image_name]
    img = cv2.imread('2015/' + image_row.image_name + '.JPG')
    # pad images at edges with 0s
    for i2, loc in not_wildebeest_sub_locs.iterrows():
        img2 = np.array(img)[max(0, loc.ycoord - 32):min(4912, loc.ycoord + 32),max(0, loc.xcoord - 32):min(7360, loc.xcoord + 32),:]
        pad_img = np.zeros((64, 64, 3))
        x_offset, y_offset = max(0, 32 - loc.xcoord), max(0, 32 - loc.ycoord)
        pad_img[y_offset:y_offset + img2.shape[0], x_offset:x_offset + img2.shape[1], :] = img2
        train_images.append(pad_img)
        train_labels.append(0)

print(len(train_images))
print(len(train_labels))

perm = list(range(len(train_images)))
random.shuffle(perm)
train_images = [train_images[index] for index in perm]
train_labels = [train_labels[index] for index in perm]

testset = int(0.1*len(train_images))

x_train = np.asarray(train_images) #[testset:,:])
y_train = np.asarray(train_labels) #[testset:,:])
x_test = np.asarray(train_images) #[:testset,:])
y_test = np.asarray(train_labels) #[:testset,:])

x_train = x_train[testset:]
y_train = y_train[testset:]
x_test = x_test[:testset]
y_test = y_test[:testset]
print(x_train.shape)
print(x_test.shape)

x_train = x_train.astype('float32')/255
x_test = x_test.astype('float32')/255

y_train = to_categorical(y_train,num_classes)
y_test = to_categorical(y_test,num_classes)

datagen = ImageDataGenerator(zoom_range=0.0, vertical_flip=False, horizontal_flip=False)
callbacks = [EarlyStopping(monitor='val_loss', patience=5, verbose=1)]

# train the model
start = time.time()
print('Training starts')
fcn_model.load_weights('vgg16-header_classifier.h5')

model_info = fcn_model.fit_generator(datagen.flow(x_train, y_train, batch_size = 1024),
                                 steps_per_epoch = 512, epochs = 200, callbacks=callbacks,
                                 validation_data = (x_test, y_test), verbose=1)
end = time.time()

print('Training ends')

fcn_model.save_weights('vgg16-header_classifier.h5')
print('Total time taken:' + str(end - start))