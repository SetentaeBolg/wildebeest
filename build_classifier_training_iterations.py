import time,os,sys
from PIL import Image
import numpy as np
import cv2
import pandas as pd
from keras.utils.np_utils import to_categorical

np.random.seed(2017)
from deepModels import getModel, getSegModel

nx = 512
ny = 512

fcnmodel = getSegModel(ny,nx)
num_classes=2

# fcnmodel.load_weights('fcn_cifar10/fcn_cifar10_weights.h5')
fcnmodel.load_weights('fcn_cifar10_weights_from_classifier.h5')
df = pd.read_csv('2015-checked-train.txt', header=None)
df2 = pd.read_csv('2015-Z-LOCATIONS.csv')
not_wildebeest = np.zeros((,3))
j = 0

for i in range(df.size):
	x_pos, y_pos = 0,0
	filename = df.iloc[i]
	print('Examining image ' + str(i) + ' of ' + str(df.size) +  ': ' + filename)
	img = Image.open('2015/' + filename + '.JPG')
	df3 = df2.loc[df2.image_name == filename]
	for y_pos in range(0, img.shape[0], ny):
		for x_pos in range(0, img.shape[1], nx):
			arr = np.array(img)
			arr = arr[y_pos:ny+y_pos,x_pos:nx+x_pos,:]
			arr = arr.transpose(1,0,2)
			arr = arr.astype('float32')/255.0
			result = fcnmodel.predict(arr)[0]
			predicted_class = result[:,:,1]
			for y_in in range(0, ny, 32):
				for x_in in range(0, nx, 32):
					flag = False
					if np.max(predicted_class[y_in:y_in+32,x_in:x_in+32]) > 0.95:
						x_t, y_t = x_pos + x_in + 15, y_pos + y_in + 15
						for d in df3:
							if math.sqrt((d.xcoord - x_t) ** 2 + (d.ycoord - y_t) ** 2) <= 130:
								flag = True
								break
						if flag = False:
							not_wildebeest[j,0] = filename
							not_wildebeest[j,1] = x_t
							not_wildebeest[j,2] = y_t
							j = j + 1

print('Finished. Saving non-wildebeest co-ordinates.')
df4 = pd.DataFrame(not_wildebeest, columns = ['image_name', 'xcoord', 'ycoord'])
df4.to_csv('new_not_wildebeest_locations.csv')

