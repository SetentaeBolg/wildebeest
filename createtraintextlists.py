import numpy as np
import pandas as pd
import os

np.random.seed(42)
df = pd.read_csv('swc_zooniverse_data_22Nov17.csv')

left = lambda x: x[:7]

image_names = [f for f in os.listdir('2015/') if os.path.isfile(os.path.join('2015', f))]
image_names = pd.DataFrame(image_names)
image_names = image_names.drop_duplicates()[0].apply(left)

print(image_names.shape[0])

image_names = image_names.iloc[np.random.permutation(image_names.shape[0])]

train = image_names[:500]
test = image_names[500:1500]

train.to_csv('train.txt', header=None, index=None, sep=' ', mode='a')
test.to_csv('test.txt', header=None, index=None, sep=' ', mode='a')