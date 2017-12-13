import pandas as pd
import time,os
from PIL import Image
import numpy as np
from keras.utils.np_utils import to_categorical
from deepModels import getModel
from keras.preprocessing.image import ImageDataGenerator
from sklearn.utils import shuffle

print('Beginning classifier training')
root_image_folder = '2015'

model = getModel()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

train_images = pd.read_csv('train.txt', header=None)
train_images.columns = ['SWC_image']
test_images = pd.read_csv('test.txt', header=None)
test_images.columns = ['SWC_image']

w_train = pd.read_csv('2015-Z-LOCATIONS.csv')
w_train = w_train[(w_train['xcoord'] > 36) & (w_train['xcoord'] < 7321) & (w_train['ycoord'] > 36) & (w_train['ycoord'] < 4875)]
w_test = w_train[w_train['image_name'].isin(list(test_images.loc[:].values.flatten()))]
w_train = w_train[w_train['image_name'].isin(list(train_images.loc[:].values.flatten()))]

nw_train = pd.read_csv('isolated_non_wildebeest.csv')
nw_test = nw_train[nw_train['image_name'].isin(list(test_images.loc[:].values.flatten()))]
nw_train = nw_train[nw_train['image_name'].isin(list(train_images.loc[:].values.flatten()))]

np.random.seed(42)
w_train = shuffle(w_train)
w_train = w_train[:500]
w_test = shuffle(w_test)
w_test = w_test[:101]
nw_train = shuffle(nw_train)
nw_train = nw_train[:500]
nw_test = shuffle(nw_test)
nw_test = nw_test[:101]

im_sz = 72
X_train = []
y_train = []
X_test = []
y_test = []

print('Acquiring images')

for index, row in w_train.iterrows():
    filename = os.path.join(root_image_folder, row['image_name'] + '.JPG')
    img = Image.open(filename)
    img = img.crop((max(row['xcoord'] - (im_sz // 2), 0), max(row['ycoord'] - (im_sz // 2), 0),
                    min(max(row['xcoord'] - (im_sz // 2), 0) + im_sz, img.size[0]),
                    min(max(row['ycoord'] - (im_sz // 2), 0) + im_sz, img.size[1])))
    arr = np.array(img)
    if arr.shape != (72,72,3):
        print(filename + ' (' + str(row['xcoord']) + ',' + str(row['ycoord']) + ') ' + str(arr.shape))
    else:
        X_train.append(arr)
        y_train.append(1)

for index, row in nw_train.iterrows():
    filename = os.path.join(root_image_folder, row['image_name'] + '.JPG')
    img = Image.open(filename)
    img = img.crop((max(row['xcoord'] - (im_sz // 2), 0), max(row['ycoord'] - (im_sz // 2), 0),
                    min(max(row['xcoord'] - (im_sz // 2), 0) + im_sz, img.size[0]),
                    min(max(row['ycoord'] - (im_sz // 2), 0) + im_sz, img.size[1])))
    arr = np.array(img)
    if arr.shape != (72,72,3):
        print(filename + ' (' + str(row['xcoord']) + ',' + str(row['ycoord']) + ') ' + str(arr.shape))
    else:
        X_train.append(arr)
        y_train.append(0)

for index, row in w_test.iterrows():
    filename = os.path.join(root_image_folder, row['image_name'] + '.JPG')
    img = Image.open(filename)
    img = img.crop((max(row['xcoord'] - (im_sz // 2), 0), max(row['ycoord'] - (im_sz // 2), 0),
                    min(max(row['xcoord'] - (im_sz // 2), 0) + im_sz, img.size[0]),
                    min(max(row['ycoord'] - (im_sz // 2), 0) + im_sz, img.size[1])))
    arr = np.array(img)
    if arr.shape != (72,72,3):
        print(filename + ' (' + str(row['xcoord']) + ',' + str(row['ycoord']) + ') ' + str(arr.shape))
    else:
        X_test.append(arr)
        y_test.append(1)

for index, row in nw_test.iterrows():
    filename = os.path.join(root_image_folder, row['image_name'] + '.JPG')
    img = Image.open(filename)
    img = img.crop((max(row['xcoord'] - (im_sz // 2), 0), max(row['ycoord'] - (im_sz // 2), 0),
                    min(max(row['xcoord'] - (im_sz // 2), 0) + im_sz, img.size[0]),
                    min(max(row['ycoord'] - (im_sz // 2), 0) + im_sz, img.size[1])))
    arr = np.array(img)
    if arr.shape != (72,72,3):
        print(filename + ' (' + str(row['xcoord']) + ',' + str(row['ycoord']) + ') ' + str(arr.shape))
    else:
        X_test.append(arr)
        y_test.append(0)

X_train = np.asarray(X_train)
y_train = np.asarray(y_train)
X_test = np.asarray(X_test)
y_test = np.asarray(y_test)
X_train = X_train.astype('float32')/255
X_test = X_test.astype('float32')/255

shuffle_index = np.random.permutation(X_train.shape[0])
X_train = X_train[shuffle_index]
y_train = y_train[shuffle_index]
y_train = to_categorical(y_train,2)
y_test = to_categorical(y_test,2)
print('Train set size : ', X_train.shape[0])
print('Test set size : ', X_test.shape[0])

datagen = ImageDataGenerator(zoom_range=0.2, vertical_flip=True,
                             horizontal_flip=True)


# train the model
start = time.time()
print('Training starts')
model_info = model.fit_generator(datagen.flow(X_train, y_train, batch_size = 128),
                                 steps_per_epoch = 512, nb_epoch = 200, 
                                 validation_data = (X_test, y_test), verbose=1)
end = time.time()

print('Training ends')

model.save_weights('cifar_weights.h5')
print('Total time taken:' + str(end - start))
