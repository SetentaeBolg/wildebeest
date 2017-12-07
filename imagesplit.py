from PIL import Image
import os

raw_image_folder = r'2015'
tiled_image_folder = raw_image_folder + '/tiled'

for file in os.listdir(raw_image_folder):
    if file.endswith('.JPG'):
        img = Image.open(raw_image_folder + '/' + file)
        for i in range(4):
            for j in range(3):
                img.crop((i * 1840, int(j * 4912/3), (i+1) * 1840, int((j+1) * 4912 / 3))).save(tiled_image_folder + '/' + file[:7] + '_' + str(i+1) + '_' + str(j+1) + '.png')