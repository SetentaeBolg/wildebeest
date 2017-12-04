from PIL import Image
import numpy as np
import pandas as pd
import scipy.misc

raw_image_folder = r'C:\Users\mc449n\Downloads\2015'
truth_image_folder = raw_image_folder + '\\truth'

data_file_csv = 'swc_zooniverse_data_22Nov17.csv'
count_data = pd.read_csv(data_file_csv)

#get a set of all unique image names in the data file
image_names = count_data.loc[:,'SWC_image']
image_names = set(image_names)

#for each image, create an array to store our ground truth
for name in image_names:
    truth = np.zeros((7360,4912), dtype=np.int16)
    newtruth = truth

    #look at rows in the datafile corresponding to that image
    count_rows = count_data.loc[count_data['SWC_image'] == name]

    #for each row, increment the array around that location by 1
    for index, row in count_rows.iterrows():
        print(index)
        tilen_horiz = int(row['tile_id'][8]) - 1
        tilen_vert = int(row['tile_id'][10]) - 1
        x_cent = int(row['xcoord'])+(tilen_horiz * 1840)
        print(x_cent)
        y_cent = int(row['ycoord'])+int(tilen_vert * int(4912/3))
        print(y_cent)
        for x in range(max(x_cent - 31,0),min(x_cent + 31,7359)):
            for y in range(max(y_cent - 31, 0),min(y_cent + 31, 4911)):
                truth[x,y] = truth[x,y] + 1
        print('row complete')
    newfilename = truth_image_folder + '\\' + name + '.png'
    scipy.misc.imsave(newfilename, truth.transpose())

    #convert the array down to 1 (wildebeest) and 0 (not wildebeest)
    for x in range(4):
        for y in range(3):
            print('Looking at tile:',x,',',y)
            #assume that if 3 counters clicked something in a given tile, it is correct
            #(if less than 3 counters worked on a given tile, they all must agree)
            no_counters = np.unique(count_rows.loc[count_rows['tile_id'] == name + '_' + str(x+1) + '_' + str(y+1)]['user_name']).size
            print('Unique counters for ' + name + '_' + str(x+1) + '_' + str(y+1) + ' = ' + str(no_counters))
            minval = max(min(no_counters,3),1)
            newtruth[x*1840:((x+1)*1840),y*int(4912/3):((y+1)*int(4912/3))] = truth[x*1840:((x+1)*1840),y*int(4912/3):((y+1)*int(4912/3))] - minval
            newtruth[x*1840:((x+1)*1840),y*int(4912/3):((y+1)*int(4912/3))] = np.heaviside(newtruth[x*1840:((x+1)*1840),y*int(4912/3):((y+1)*int(4912/3))], 0)

    #save out image
    newfilename = truth_image_folder + '\\' + name + '.png'
    newnewfilename = truth_image_folder + '\\' + name + '_amended.png'
    #scipy.misc.imsave(newfilename, truth.transpose())
    scipy.misc.imsave(newnewfilename, newtruth.transpose())
    break
