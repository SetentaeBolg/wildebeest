import numpy as np
import pandas as pd
from PIL import Image
import glob
import itertools


class SegDataGen2(object):

	def __init__(self, file_path = None, dim_x = 2000, dim_y = 2000, batch_size = 16):
		self.file_path = file_path
		self.dim_x = int(dim_x)
		self.dim_y = int(dim_y)
		self.batch_size = int(batch_size)
		self.x_pos = 0
		self.y_pos = 0


	def getImageArr( self, path):
		width = self.dim_x
		height = self.dim_y
		left = self.x_pos
		top = self.y_pos

		try:
			img = Image.open(path)
			img = img.crop((left,
				top,
				left + width,
				top + height))
			img = np.array(img).astype(np.float32)
			img = img/255.0
			assert img.shape == (width, height, 3)

			return img
		except Exception as e:
			print(e)
			img = np.zeros((  width , height  , 3 ))
			return img





	def getSegmentationArr( self, path , nClasses):
		width = self.dim_x
		height = self.dim_y
		left = self.x_pos
		top = self.y_pos

		seg_labels = np.zeros((  width , height  , nClasses ))
		try:
			img = Image.open(path)
			img = img.crop((left,
				top,
				left + width,
				top + height))
			img = np.array(img)[:, : , 0]
			#correct for pixel intensity
			img = np.around((img * (nClasses - 1)) / 255.0)

			for c in range(nClasses):
				seg_labels[: , : , c ] = (img == c ).astype(int)

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

		zipped = itertools.cycle( zip(images,segmentations) )
		im = ''
		seg = ''
		im_x, im_y = np.array(Image.open(df.iloc[0].values[0])).shape[0:2]

		while True:
			X = []
			Y = []
			flag = True
			i = 0
			while flag:
				if self.y_pos + self.dim_y > im_y or im == '':
					im , seg = zipped.__next__()
					self.x_pos = 0
					self.y_pos = 0
				if df is None:
					i = i + 1
					X.append( self.getImageArr(str(im)))
					Y.append( self.getSegmentationArr( str(seg) , n_classes )  )
				elif im in list(df.values.flatten()):
					i = i + 1
					X.append( self.getImageArr(str(im) )  )
					Y.append( self.getSegmentationArr( str(seg) , n_classes )  )
				self.x_pos = self.x_pos + self.dim_x
				if self.x_pos + self.dim_x > im_x:
					self.x_pos = 0
					self.y_pos = self.y_pos + self.dim_y
				if i >= self.batch_size:
					flag = False
			X = np.transpose(np.array(X), (0,2,1,3))
			Y = np.transpose(np.array(Y), (0,2,1,3))
			# print(X.shape, Y.shape)
			yield X, Y
