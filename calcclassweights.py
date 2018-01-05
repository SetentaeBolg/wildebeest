import pandas as pd
import numpy as np
from PIL import Image

trainfile = '2015-checked-train.txt'
testfile = '2015-checked-test.txt'

df = pd.read_csv(trainfile, header=None)

numclasses = 2

classcounts = np.zeros((numclasses))
totalcounts = np.zeros((numclasses))

for t in df.values.flatten():
	img = Image.open('2015/truth/' + t + '.png')
	arr = np.asarray(img)[:,:,0] / 255.0
	unique, counts = np.unique(arr, return_counts = True)
	print(t + ':' + str(dict(zip(unique,counts))))
	for i in range(numclasses):
		classcounts[i] = classcounts[i] + (0 if counts.shape[0] < i+1 else counts[i])
		totalcounts[i] = totalcounts[i] + (0 if counts.shape[0] < i+1 else np.sum(counts))

freq = np.zeros((numclasses))
classweight = np.zeros((numclasses))

for i in range(numclasses):
	freq[i] = classcounts[i] / totalcounts[i]

medfreq = np.median(freq)

for i in range(numclasses):
	classweight[i] = medfreq / freq[i]

print(classweight)
