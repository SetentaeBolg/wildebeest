import numpy as np
import matplotlib.pyplot as plt
from pylab import *
import os
import sys
import pickle
from keras.optimizers import SGD, Adam, Nadam
from keras.callbacks import *
from keras.objectives import *
from keras.metrics import binary_accuracy
from keras.models import load_model
import keras.backend as K
#import keras.utils.visualize_util as vis_util

from deepModels import *
from segdatagenerator import *
import time
import tensorflow as tf


def sparse_accuracy_ignoring_last_label(y_true, y_pred):
    nb_classes = K.int_shape(y_pred)[-1]
    y_pred = K.reshape(y_pred, (-1, nb_classes))

    y_true = K.one_hot(tf.to_int32(K.flatten(y_true)),
                       nb_classes + 1)
    unpacked = tf.unstack(y_true, axis=-1)
    legal_labels = ~tf.cast(unpacked[-1], tf.bool)
    y_true = tf.stack(unpacked[:-1], axis=-1)

    return K.sum(tf.to_float(legal_labels & K.equal(K.argmax(y_true, axis=-1), K.argmax(y_pred, axis=-1)))) / K.sum(tf.to_float(legal_labels))


# Softmax cross-entropy loss function for pascal voc segmentation
# and models which do not perform softmax.
# tensorlow only
def softmax_sparse_crossentropy_ignoring_last_label(y_true, y_pred):
    y_pred = K.reshape(y_pred, (-1, K.int_shape(y_pred)[-1]))
    log_softmax = tf.nn.log_softmax(y_pred)

    y_true = K.one_hot(tf.to_int32(K.flatten(y_true)), K.int_shape(y_pred)[-1]+1)
    unpacked = tf.unstack(y_true, axis=-1)
    y_true = tf.stack(unpacked[:-1], axis=-1)

    cross_entropy = -K.sum(y_true * log_softmax, axis=1)
    cross_entropy_mean = K.mean(cross_entropy)

    return cross_entropy_mean


# Softmax cross-entropy loss function for coco segmentation
# and models which expect but do not apply sigmoid on each entry
# tensorlow only
def binary_crossentropy_with_logits(ground_truth, predictions):
    return K.mean(K.binary_crossentropy(ground_truth,
                                        predictions,
                                        from_logits=True),
                  axis=-1)


def train(batch_size, epochs, lr_base, lr_power, weight_decay, classes,
          model_name, train_file_path, val_file_path,
          data_dir, label_dir, target_size=None, batchnorm_momentum=0.9,
          resume_training=False, class_weight=None, dataset='VOC2012',
          loss_fn = softmax_sparse_crossentropy_ignoring_last_label,
          metrics = [sparse_accuracy_ignoring_last_label],
          loss_shape=None,
          label_suffix='.png',
          data_suffix='.jpg',
          ignore_label=255,
          label_cval=255):
    if target_size:
        input_shape = target_size + (3,)
    else:
        input_shape = (None, None, 3)
    batch_shape = (batch_size,) + input_shape

    ###########################################################
    current_dir = os.path.dirname(os.path.realpath(__file__))
    save_path = os.path.join(current_dir, model_name)
    # ###############learning rate scheduler####################
    def lr_scheduler(epoch, mode='power_decay'):

        if mode is 'power_decay':
            # original lr scheduler
            lr = lr_base * ((1 - float(epoch)/epochs) ** lr_power)
        if mode is 'exp_decay':
            # exponential decay
            lr = (float(lr_base) ** float(lr_power)) ** float(epoch+1)
        # adam default lr
        if mode is 'adam':
            lr = 0.001

        if mode is 'progressive_drops':
            # drops as progression proceeds, good for sgd
            if epoch > 0.9 * epochs:
                lr = 0.0001
            elif epoch > 0.75 * epochs:
                lr = 0.001
            elif epoch > 0.5 * epochs:
                lr = 0.01
            else:
                lr = 0.1

        print('lr: %f' % lr)
        return lr

    scheduler = LearningRateScheduler(lr_scheduler)

    checkpoint_path = os.path.join(save_path, 'checkpoint_weights.h5')

    model = getSegModel(input_width=input_shape[0], input_height=input_shape[1])

    optimizer = SGD(lr=lr_base, momentum=0.9)

    model.compile(loss=loss_fn,
                  optimizer=optimizer,
                  metrics=metrics)

    if resume_training:
        model.load_weights(checkpoint_path, by_name=True)
    else:
        model.load_weights('fcn_cifar10_weights.h5')

    model_path = os.path.join(save_path, "model.json")

    # save model structure
    f = open(model_path, 'w')
    model_json = model.to_json()
    f.write(model_json)
    f.close

    model.summary()

    callbacks = [scheduler]

    # ####################### tfboard ###########################
    tensorboard = TensorBoard(log_dir=os.path.join(save_path, 'logs'), histogram_freq=10, write_graph=True)
    callbacks.append(tensorboard)
    # ################### checkpoint saver#######################
    checkpoint = ModelCheckpoint(filepath=os.path.join(save_path, 'checkpoint_weights.h5'), save_weights_only=True)#.{epoch:d}
    callbacks.append(checkpoint)
    # set data generator and train
    train_datagen = SegDataGenerator(zoom_range=[0.5, 2.0],
                                     zoom_maintain_shape=True,
                                     crop_mode='random',
                                     crop_size=target_size,
                                     rotation_range=0.,
                                     shear_range=0,
                                     horizontal_flip=True,
                                     channel_shift_range=20.,
                                     fill_mode='constant',
                                     label_cval=label_cval)

    def get_file_len(file_path):
        fp = open(file_path)
        lines = fp.readlines()
        fp.close()
        return len(lines)

    # from Keras documentation: Total number of steps (batches of samples) to yield from generator before declaring one epoch finished
    # and starting the next epoch. It should typically be equal to the number of unique samples of your dataset divided by the batch size.
    steps_per_epoch = int(np.ceil(get_file_len(train_file_path) / float(batch_size)))

    history = model.fit_generator(
        generator=train_datagen.flow_from_directory(
            file_path=train_file_path,
            data_dir=data_dir, data_suffix=data_suffix,
            label_dir=label_dir, label_suffix=label_suffix,
            classes=classes,
            target_size=target_size, color_mode='rgb',
            batch_size=batch_size, shuffle=True,
            loss_shape=loss_shape,
            ignore_label=ignore_label,
            # save_to_dir='Images/'
        ),
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        callbacks=callbacks,
        workers=4,
        # validation_data=val_datagen.flow_from_directory(
        #     file_path=val_file_path, data_dir=data_dir, data_suffix='.jpg',
        #     label_dir=label_dir, label_suffix='.png',classes=classes,
        #     target_size=target_size, color_mode='rgb',
        #     batch_size=batch_size, shuffle=False
        # ),
        # nb_val_samples = 64
        class_weight=class_weight
       )

    model.save_weights(save_path+'/fcn_cifar10_weights.h5')

model_name = 'fcn_cifar10'
batch_size = 16
batchnorm_momentum = 0.95
epochs = 250
lr_base = 0.01 * (float(batch_size) / 16)
lr_power = 0.9
resume_training = False
weight_decay = 1e-4
target_size = (320, 320)
dataset = 'DIY'

#train_file_path = os.path.expanduser('~/.keras/datasets/VOC2012/combined_imageset_train.txt')
train_file_path = os.path.expanduser('2015-checked-train.txt') #Data/VOClarge/VOC2012/ImageSets/Segmentation
#val_file_path   = os.path.expanduser('~/.keras/datasets/VOC2012/combined_imageset_val.txt')
data_dir        = os.path.expanduser('2015')
label_dir       = os.path.expanduser('2015/truth')
data_suffix='.jpg'
label_suffix='.png'
classes = 2

# ###################### loss function & metric ########################
loss_fn = softmax_sparse_crossentropy_ignoring_last_label
metrics = [sparse_accuracy_ignoring_last_label]
loss_shape = None
ignore_label = 255
label_cval = 255

# Class weight is not yet supported for 3+ dimensional targets
# class_weight = {i: 1 for i in range(classes)}
# # The background class is much more common than all
# # others, so give it less weight!
# class_weight[0] = 0.1
class_weight = None

config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
session = tf.Session(config=config)
K.set_session(session)
train(batch_size, epochs, lr_base, lr_power, weight_decay, classes, model_name, train_file_path, '',
      data_dir, label_dir, target_size=target_size, batchnorm_momentum=batchnorm_momentum, resume_training=resume_training,
      class_weight=class_weight, loss_fn=loss_fn, metrics=metrics, loss_shape=loss_shape, data_suffix=data_suffix,
      label_suffix=label_suffix, ignore_label=ignore_label, label_cval=label_cval)
