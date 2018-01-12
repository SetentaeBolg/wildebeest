import os
import cv2
import numpy as np
import pandas as pd
import itertools
from sklearn.utils import shuffle
from PIL import Image

np.random.seed(42)

root_image_folder = '2015'
#small_image_folder = os.path.join(root_image_folder, 'small_images')

# generate classifier images

df = pd.read_csv('2015-checked-train.txt', header=None)
df.columns = ['SWC_image']

df13 = pd.read_csv('2015-checked-test.txt', header=None)
df13.columns = ['SWC_image']

# df = df.append(df13)

df2 = pd.read_csv('2015-Z-LOCATIONS.csv')

df2 = df2[df2['image_name'].isin(list(df.loc[:].values.flatten()))]

df = df[df.isin(list(df2['image_name'].values.flatten()))]
df = df.replace('', np.nan)
df = df.dropna()

df.to_csv('2015-checked-train-reduced.txt', header=None, index=False, mode='w')

# listx = list(range(0, 7359, 1440))
# listy = list(range(0, 4911, 1440))

# listimg = list(df.loc[:].values.flatten())
# df5 = pd.DataFrame(list(itertools.product(listimg,listx,listy)))

# df5.columns = ['image_name', 'xcoord', 'ycoord']

# indexes = []

# for index, row in df5.iterrows():
#     df3 = df2[df2['image_name'] == row['image_name']]
#     flag = True
#     if df3.shape[0] > 0:
#         for index2, row2 in df3.iterrows():
#             if np.sqrt((row2['xcoord'] - row['xcoord'])**2 + (row2['ycoord'] - row['ycoord'])**2) < 144:
#                 flag = False
#                 break
#     if flag == True:
#         indexes.append(index)

# df6 = df5.loc[indexes]

# df6.to_csv('isolated_non_wildebeest.csv', index=False, mode='w')

# crop out images and save in relevant folders for later cifar training

#w_train = pd.read_csv('2015-Z-LOCATIONS.csv')
#w_train = w_train[w_train['image_name'].isin(list(df.loc[:].values.flatten()))]

#nw_train = pd.read_csv('isolated_non_wildebeest.csv')
#nw_train = nw_train[nw_train['image_name'].isin(list(df.loc[:].values.flatten()))]

#w_train = shuffle(w_train)
#w_train = w_train[:500]
#nw_train = shuffle(nw_train)
#nw_train = nw_train[:500]

#im_sz = 72
#X_train = []
#y_train = []
#X_test = []
#y_test = []

#for index, row in w_train.iterrows():
#    filename = os.path.join(root_image_folder, row['image_name'] + '.JPG')
#    img = Image.open(filename)
#    img = img.crop((max(row['xcoord'] - (im_sz // 2), 0), max(row['ycoord'] - (im_sz // 2), 0),
#                    min(max(row['xcoord'] - (im_sz // 2), 0) + im_sz, img.size[0]),
#                    min(max(row['ycoord'] - (im_sz // 2), 0) + im_sz, img.size[1])))
#    img.save(small_image_folder + '/wildebeest/' + row['image_name'] + '_' + str(index) + '.png')

#for index, row in nw_train.iterrows():
#    filename = os.path.join(root_image_folder, row['image_name'] + '.JPG')
#    img = Image.open(filename)
#    img = img.crop((max(row['xcoord'] - (im_sz // 2), 0), max(row['ycoord'] - (im_sz // 2), 0),
#                    min(max(row['xcoord'] - (im_sz // 2), 0) + im_sz, img.size[0]),
#                    min(max(row['ycoord'] - (im_sz // 2), 0) + im_sz, img.size[1])))
#    img.save(small_image_folder + '/nonwildebeest/' + row['image_name'] + '_' + str(index) + '.png')
