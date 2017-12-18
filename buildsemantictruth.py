from PIL import Image, ImageDraw
import numpy as np
import pandas as pd
import os

raw_image_folder = '2015'
truth_image_folder = raw_image_folder + '/truth'

data_file_csv = '2015-Z-LOCATIONS.csv'
count_data = pd.read_csv(data_file_csv)

#get a set of all unique image names in the data file
image_names = os.listdir(raw_image_folder)

#for each image, create an array to store our ground truth
for name in image_names:
    if name.endswith('.JPG'):

        fname=name[:7]

        #look at rows in the datafile corresponding to that image
        count_rows = count_data.loc[count_data['image_name'] == fname]

        img = Image.new(mode='RGB',size=(7360,4912),color=(0,0,0))
        draw = ImageDraw.Draw(img,mode='RGB')

        #for each row, increment the array around that location by 1
        for index, row in count_rows.iterrows():
            draw.ellipse((int(row.xcoord) - 15, int(row.ycoord) - 15, int(row.xcoord) + 15, int(row.ycoord) + 15), fill=(255,255,255))
        newfilename = truth_image_folder + '/' + fname + '.png'
        img.save(newfilename)
