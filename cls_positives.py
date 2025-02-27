
import numpy as np
import pandas as pd
import os
import cv2


ROOTDIR = ''
image_dir = ROOTDIR + '2015/'
train_dir = image_dir + 'cls_train_images/w/'

allfile = ROOTDIR  + '2015-Z-LOCATIONS.csv'
w_train = pd.read_csv(allfile)

train_images = np.genfromtxt(ROOTDIR + '2015-checked-train.txt',dtype='str')


im_size=64
sz_2=im_size//2

w_train = w_train[(w_train['xcoord'] > 32) & (w_train['xcoord'] < 7325) & (w_train['ycoord'] > 32) & (w_train['ycoord'] < 4879)]

for imagename in train_images: 
    im = cv2.imread(image_dir + imagename + '.JPG')
    df = w_train[w_train['image_name']==imagename]

    for i,point in df.iterrows():


        img = im[int(point['ycoord']) - sz_2:int(point['ycoord'])+sz_2,int(point['xcoord']) - sz_2:int(point['xcoord'])+sz_2,:]

        if img.shape == (im_size,im_size,3):
            cv2.imwrite(train_dir + imagename + '_' + str(i) + '.jpg', img)

