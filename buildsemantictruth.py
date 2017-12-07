from PIL import Image
import os
import numpy as np
import pandas as pd
import scipy.misc

raw_image_folder = '2015'
tiled_image_folder = raw_image_folder + '/tiled'
tiled_truth_image_folder = tiled_image_folder + '/truth'
truth_image_folder = raw_image_folder + '/truth'

data_file_csv = 'swc_zooniverse_cluster_found_coords.csv'
count_data = pd.read_csv(data_file_csv)

#get a set of all unique image names in the data file
image_names = count_data.loc[:,'tile_id']
image_names = set(image_names)
'''
#for each image, create an array to store our ground truth
for name in image_names:

    truth = np.zeros((1840,1638), dtype=np.int16)

    #look at rows in the datafile corresponding to that image
    count_rows = count_data.loc[count_data['tile_id'] == name]

    #for each row, increment the array around that location by 1
    for index, row in count_rows.iterrows():
        x_cent = int(row['xcoord'])
        y_cent = int(row['ycoord'])
        if x_cent > 1839 or y_cent > 1637:
            print ('Correcting coords from (' + str(x_cent) + ', ' + str(y_cent) + ') to (' + str(max(x_cent, 1839)) + ', ' + str(max(y_cent, 1637)) + ')')
        for x in range(max(x_cent - 31,0),min(x_cent + 31,1839)):
            for y in range(max(y_cent - 31, 0),min(y_cent + 31, 1637)):
                truth[x,y] = 1
    newfilename = tiled_truth_image_folder + '/' + name + '.png'
    scipy.misc.imsave(newfilename, truth.transpose())
'''
#stitch together images
image_names = list(image_names)
for i in range(len(image_names)):
    image_names[i] = image_names[i][:7]
image_names = set(image_names)
for name in image_names:
    print(name)
    image_location = tiled_truth_image_folder + '/' + name
    list_o_names = []
    for i in range(3):
        for j in range(4):
            list_o_names.append(image_location + '_' + str(j+1) + '_' + str(i+1) + '.png')
    images = []
    for name2 in list_o_names:
        if os.path.isfile(name2):
            images.append(Image.open(name2))
        else:
            if name2.endswith('3.png'):
                images.append(Image.new('RGB',(1840,1638)))
            else:
                images.append(Image.new('RGB', (1840, 1637)))
    new_im = Image.new('RGB',(7360,4912))
    for i in range(3):
        for j in range(4):
            new_im.paste(images[i*4+j], (j*1840,int(i*(4912/3))))
    new_im.save(truth_image_folder + '/' + name + '.png')
