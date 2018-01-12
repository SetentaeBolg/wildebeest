from segdatagenerator import *
import numpy as np
import os
import sys
import pickle
from keras.optimizers import SGD, Adam, Nadam
from keras.callbacks import *
from keras.objectives import *
from keras.metrics import binary_accuracy
from keras.models import load_model
from keras.losses import categorical_crossentropy
import keras.backend as K
from deepModels import *
import time
import tensorflow as tf

#loss function
def weighted_categorical_crossentropy_fcn_loss(y_true, y_pred):
    # y_true is a matrix of weight-hot vectors (like 1-hot, but they have weights instead of 1s)
    y_true_mask = K.sign(y_true)  # [0 0 W 0] -> [0 0 1 0] where W > 0.
    cce = categorical_crossentropy(y_true_mask, y_pred)  # one dim less (each 1hot vector -> float number)
    y_true_weights_maxed = K.max(y_true, axis=-1)  # [0 120 0 0] -> 120 - get weight for each weight-hot vector
    wcce = cce * y_true_weights_maxed
    return K.sum(wcce)

model_name = 'fcn_cifar10'
batch_size = 32
epochs = 250
lr_base = 0.01 * (float(batch_size) / 16)
target_size = (512, 512)
class_weights = [0.50038, 666.47713]

train_file_path = os.path.expanduser('2015-checked-train-reduced.txt')
val_file_path   = os.path.expanduser('2015-checked-test.txt')
data_dir        = os.path.expanduser('2015/')
label_dir       = os.path.expanduser('2015/truth/')
data_suffix     = '.JPG'
label_suffix    = '.png'
classes = 2

# ###################### loss function & metric ########################
loss_fn = weighted_categorical_crossentropy_fcn_loss
metrics = ['accuracy']

config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
session = tf.Session(config=config)
K.set_session(session)


input_shape = target_size + (3,)
batch_shape = (batch_size,) + input_shape

###########################################################
current_dir = os.path.dirname(os.path.realpath(__file__))
save_path = os.path.join(current_dir, model_name)

checkpoint_path = os.path.join(save_path, 'checkpoint_weights.h5')

model = getSegModel(input_width=target_size[0], input_height=target_size[1])

# comment out to refresh model
# model.load_weights(checkpoint_path)
model.load_weights('fcn_cifar10_weights_from_classifier.h5')

optimizer = Nadam()

model.compile(loss=loss_fn,
              optimizer=optimizer,
              metrics=metrics)

model.summary()

# ################### checkpoint saver ######################
checkpoint = ModelCheckpoint(filepath=os.path.join(save_path, 'checkpoint_weights.h5'), save_weights_only=True)#.{epoch:d}
callbacks = [checkpoint]

# ################### early stopping ########################
earlystopping = EarlyStopping(monitor='val_loss', patience=6, verbose=1)
callbacks.append(earlystopping)

# set data generator and train

def get_file_len(file_path):
    fp = open(file_path)
    lines = fp.readlines()
    fp.close()
    return len(lines)

# from Keras documentation: Total number of steps (batches of samples) to yield from generator before declaring one epoch finished
# and starting the next epoch. It should typically be equal to the number of unique samples of your dataset divided by the batch size.
#
# As we are using samples taken from tiles of images and a certain similarity within an image is a reasonable assumption,
# I am reducing the steps per epoch to a lower amount
steps_per_epoch = 200 #int(np.ceil(((7360 // target_size[0]) * (4912 // target_size[1]) * get_file_len(train_file_path)) / float(batch_size)))

training_generator = SegDataGen2(train_file_path,target_size[0],target_size[1],batch_size, class_weights).generate(data_dir,label_dir,classes)
test_generator = SegDataGen2(val_file_path,target_size[0],target_size[1],batch_size, class_weights).generate(data_dir,label_dir,classes)

history = model.fit_generator(
    generator=training_generator,
    steps_per_epoch = steps_per_epoch,
    epochs = epochs,
    callbacks = callbacks,
    validation_steps = steps_per_epoch // 5, #int(np.ceil(((7360 // target_size[0]) * (4912 // target_size[1]) * get_file_len(train_file_path)) / float(batch_size))), 
    validation_data = test_generator
    )

model.save_weights(save_path+'/fcn_cifar10_weights.h5')
