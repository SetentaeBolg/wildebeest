import pandas as pd
import time,os
from PIL import Image
import numpy as np
from keras.utils.np_utils import to_categorical
from keras.callbacks import EarlyStopping
from deepModels import getModel
from keras.preprocessing.image import ImageDataGenerator
from sklearn.utils import shuffle
from segdatagenerator import *

print('Beginning segmentation training')
root_image_folder = '2015'

model = getModel()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

train_images = pd.read_csv('train.txt', header=None)
train_images.columns = ['SWC_image']
test_images = pd.read_csv('test.txt', header=None)
test_images.columns = ['SWC_image']

datagen = SegDataGenerator(zoom_range=0.2, vertical_flip=True,
                             horizontal_flip=True)


callbacks = [EarlyStopping(monitor='val_loss', patience=2, verbose=1)]

# train the model
start = time.time()
print('Training starts')
model_info = model.fit_generator(datagen.flow(X_train, y_train, batch_size = 128),
                                 steps_per_epoch = 512, epochs = 200, callbacks=callbacks,
                                 validation_data = (X_test, y_test), verbose=1)
end = time.time()

print('Training ends')

model.save_weights('cifar_weights.h5')
print('Total time taken:' + str(end - start))
