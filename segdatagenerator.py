import numpy as np
import pandas as pd
from PIL import Image
import glob
import itertools
import random


class SegDataGen2(object):

	def __init__(self, file_path = None, dim_x = 2000, dim_y = 2000, batch_size = 16, class_weight = None):
		self.file_path = file_path
		self.dim_x = int(dim_x)
		self.dim_y = int(dim_y)
		self.batch_size = int(batch_size)
		self.class_weight = class_weight


	def getImageArr(self, path, x_pos, y_pos):
		width = self.dim_x
		height = self.dim_y
		left = x_pos * self.dim_x
		top = y_pos * self.dim_y

		try:
			img = Image.open(path)
			img = img.crop((left,
				top,
				left + width,
				top + height))
			#img.save('sdgimg.png')
			img = np.array(img).astype(np.float32)
			img = img/255.0
			assert img.shape == (width, height, 3)

			return img

		except Exception as e:
			print(e)
			img = np.zeros((  width , height  , 3 ))
			return img


	def getSegmentationArr(self, path , nClasses, x_pos, y_pos):
		width = self.dim_x
		height = self.dim_y
		left = x_pos * self.dim_x
		top = y_pos * self.dim_y

		seg_labels = np.zeros((  width , height  , nClasses ))
		try:
			img = Image.open(path)
			img = img.crop((left,
				top,
				left + width,
				top + height))
			#img.save('sdgseg.png')
			img = np.array(img)[:, : , 0]
			#print(np.sum(img[:,:]))
			#correct for pixel intensity
			img = np.around((img * (nClasses - 1)) / 255.0)

			for c in range(nClasses):
				seg_labels[: , : , c ] = ((img == c ).astype(int) * (1 if self.class_weight is None else self.class_weight[c]))

		except Exception as e:
			print (e)
		
		seg_labels = np.reshape(seg_labels, ( width, height , nClasses ))
		return seg_labels


	def generate( self, images_path , segs_path ,  n_classes ):
	
		assert images_path[-1] == '/'
		assert segs_path[-1] == '/'
		if self.file_path:
			df = pd.read_csv(self.file_path, header=None)
			df = images_path + df
			df = df + '.JPG'

		images = glob.glob( images_path + "*.JPG"  ) + glob.glob( images_path + "*.png"  ) +  glob.glob( images_path + "*.jpeg"  )
		images.sort()
		segmentations  = glob.glob( segs_path + "*.JPG"  ) + glob.glob( segs_path + "*.png"  ) +  glob.glob( segs_path + "*.jpeg"  )
		segmentations.sort()

		assert len( images ) == len(segmentations)
		for im , seg in zip(images,segmentations):
			assert(  im.split('/')[-1].split(".")[0] ==  seg.split('/')[-1].split(".")[0] )

		if self.file_path:
			for im in images:
				if not im in list(df.values.flatten()):
					del segmentations[images.index(im)]
					del images[images.index(im)]

		zipped = zip(images,segmentations)
		im = ''
		seg = ''
		im_x, im_y = np.array(Image.open(df.iloc[0].values[0])).shape[0:2]
		x_range = im_x // self.dim_x
		y_range = im_y // self.dim_y
		prod = list(itertools.product(list(zipped), list(range(x_range)), list(range(y_range))))
		random.shuffle(prod)
		zipped = itertools.cycle(prod)
		while True:
			X = []
			Y = []
			flag = True
			i = 0
			while flag:
				(im , seg), x_pos, y_pos = zipped.__next__()
				#print(im,seg,str(x_pos),str(y_pos))
				i = i + 1
				X.append(self.getImageArr(str(im), x_pos, y_pos))
				Y.append(self.getSegmentationArr(str(seg), n_classes, x_pos, y_pos))
				if i >= self.batch_size:
					flag = False
			X = np.transpose(np.array(X), (0,2,1,3))
			Y = np.transpose(np.array(Y), (0,2,1,3))
			yield X, Y
