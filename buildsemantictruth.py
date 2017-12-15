from PIL import Image
import numpy as np
import pandas as pd

raw_image_folder = '2015'
truth_image_folder = raw_image_folder + '/truth'

data_file_csv = '2015-Z-LOCATIONS.csv'
count_data = pd.read_csv(data_file_csv)

#get a set of all unique image names in the data file
image_names = count_data.loc[:,'image_name']
image_names = set(image_names)

#for each image, create an array to store our ground truth
for name in image_names:

    truth = np.zeros((7360,4912), dtype=np.int16)

    #look at rows in the datafile corresponding to that image
    count_rows = count_data.loc[count_data['image_name'] == name]

    #for each row, increment the array around that location by 1
    for index, row in count_rows.iterrows():
        x_cent = int(row['xcoord'])
        y_cent = int(row['ycoord'])
        for x in range(max(x_cent - 31,0),min(x_cent + 31,7359)):
            for y in range(max(y_cent - 31, 0),min(y_cent + 31, 4911)):
                truth[x,y] = 1
    newfilename = truth_image_folder + '/' + name + '.png'
    img = Image.fromarray(truth.transpose())
    img.save(newfilename)