import os
import cv2
import numpy as np
import pandas as pd
import itertools

# generate classifier images

df = pd.read_csv('train.txt', header=None)
df.columns = ['SWC_image']

df2 = pd.read_csv('swc_zooniverse_cluster_found_coords_xycorrected.csv')

df2 = df2[df2['image_name'].isin(list(df.loc[:].values.flatten()))]

indexes = []

for index, row in df2.iterrows():
    df3 = df2[df2['image_name'] == row['image_name']]
    df3 = df3[df3.index != index]
    flag = True
    if df3.shape[0] > 0:
        for index2, row2 in df3.iterrows():
            if np.sqrt((row2['xcoord'] - row['xcoord'])**2 + (row2['ycoord'] - row['ycoord'])**2) < 144:
                flag = False
                break
    if flag == True:
        indexes.append(index)

df4 = df2.loc[indexes]

df4.to_csv('isolated_wildebeest.csv', index=False, mode='w')

listx = list(range(0, 7359, 1440))
listy = list(range(0, 4911, 1440))
listimg = list(df.loc[:].values.flatten())
df5 = pd.DataFrame(list(itertools.product(listimg,listx,listy)))

df5.columns = ['image_name', 'xcoord', 'ycoord']

indexes = []

for index, row in df5.iterrows():
    df3 = df2[df2['image_name'] == row['image_name']]
    flag = True
    if df3.shape[0] > 0:
        for index2, row2 in df3.iterrows():
            if np.sqrt((row2['xcoord'] - row['xcoord'])**2 + (row2['ycoord'] - row['ycoord'])**2) < 144:
                flag = False
                break
    if flag == True:
        indexes.append(index)

df6 = df5.loc[indexes]

df6.to_csv('isolated_non_wildebeest.csv', index=False, mode='w')