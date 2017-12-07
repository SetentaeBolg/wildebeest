import numpy as np
import pandas as pd

np.random.seed(42)
df = pd.read_csv('swc_zooniverse_data_22Nov17.csv')

image_names = df.loc[:, ['SWC_image']]
image_names = image_names.drop_duplicates()

image_names = image_names.iloc[np.random.permutation(image_names.shape[0])]

train = image_names[:500]
test = image_names[500:1000]

train.to_csv('train.txt', header=None, index=None, sep=' ', mode='a')
test.to_csv('test.txt', header=None, index=None, sep=' ', mode='a')