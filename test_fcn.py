import time,os,sys
from PIL import Image
import numpy as np
import cv2
from keras.utils.np_utils import to_categorical

np.random.seed(2017)
from deepModels import *

nx = 512
ny = 512

fcnmodel = getVgg16SegModel(ny,nx)


num_classes=2
# fcnmodel.load_weights('fcn_cifar10/fcn_cifar10_weights.h5')
fcnmodel.load_weights('vgg16-header_classifier.h5')

# test FCN on section of survey image
X = []
# filename = "2015/SWC1717.JPG"
filename = '2015/test/test_in.png'
img = Image.open(filename)
arr = np.array(img)
# arr = arr[500:ny+500,1000:nx+1000,:]
X.append(arr)
img = Image.fromarray(X[0])
img.save('2015/test/input.png')

X = np.asarray(X)
X = X.astype('float32')/255

result = fcnmodel.predict(X)[0]
predicted_class = np.argmax(result, axis=2)
outRGB = cv2.cvtColor(255 * predicted_class.astype(np.uint8) // (num_classes - 1) , cv2.COLOR_GRAY2BGR)
cv2.imwrite('2015/test/test.png',outRGB)
# predicted_class = result[:,:,1]
# predicted_class = predicted_class - 0.9
# predicted_class[predicted_class < 0] = 0
# predicted_class = predicted_class * 10
# outRGB = cv2.cvtColor(255 * predicted_class, cv2.COLOR_GRAY2BGR)
kernel = np.ones((3,3), np.uint8)
outRGB = cv2.erode(outRGB, kernel, iterations = 7)

# print(np.max(outRGB))
        
cv2.imwrite('2015/test/test_eroded.png',outRGB)
