#not quite working yet, fine tuning required

from PIL import Image
import numpy as np
import pandas as pd
import scipy.misc

raw_image_folder = r'C:\Users\mc449n\Downloads\2015'
truth_image_folder = r'C:\Users\mc449n\Downloads\2015\truth'

data_file_csv = 'swc_zooniverse_data_22Nov17.csv'

count_data = pd.read_csv(data_file_csv)

image_names = count_data.loc[:,'SWC_image']

image_names = set(image_names)

def dist(x1,y1,x2,y2):
    return np.sqrt((x1-x2)**2 + (y1-y2)**2)

for name in image_names:
    truth = np.zeros((7360,4912), dtype=np.uint16)
    count_rows = count_data.loc[count_data['SWC_image'] == name]
    for index, row in count_rows.iterrows():
        print(index)
        tilen_horiz = int(row['tile_id'][8]) - 1
        tilen_vert = int(row['tile_id'][10]) - 1
        x_cent = int(row['xcoord'])+(tilen_horiz * int(4912/3))
        print(x_cent)
        y_cent = 4912 - int(row['ycoord'])+int(tilen_horiz * 1840)
        print(y_cent)
        for x in range(max(x_cent - 31,0),min(x_cent + 31,4911)):
            for y in range(max(y_cent - 31, 0),min(y_cent + 31, 7359)):
                truth[y,x] = truth[y,x] + 1
        print('row complete')
    newfilename = truth_image_folder + '\\' + name + '.png'
    scipy.misc.imsave(newfilename, truth.transpose())
    break
