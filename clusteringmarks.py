import numpy as np
import pandas as pd
import sklearn.cluster as skc

data_file_csv = 'swc_zooniverse_data_22Nov17.csv'
count_data = pd.read_csv(data_file_csv)
data = []

for tile in count_data['tile_id'].drop_duplicates():
    df = count_data.loc[count_data['tile_id'] == tile].dropna().groupby('user_name')['mark_index'].nunique()
    if len(df.index) > 0:
        accepted_number_of_marks = max(df)
    else:
        accepted_number_of_marks = 0
    print('Tile: ' + tile + ', accepted marks: ' + str(accepted_number_of_marks))
    if accepted_number_of_marks >= 2:
        model = skc.AgglomerativeClustering(accepted_number_of_marks)
        X = count_data.loc[count_data['tile_id'] == tile][['xcoord','ycoord']].dropna(axis=0)
        y = model.fit_predict(X)
        X['cluster'] = y
        X = X.groupby('cluster').filter(lambda x: len(x) >= 5)
        X = X.groupby('cluster').first()
        X['tile_id'] = tile
        for index, row in X.iterrows():
            data.append([row.tile_id,index,row.xcoord,row.ycoord])
    elif accepted_number_of_marks == 1:
        for index, row in count_data.loc[count_data['tile_id'] == tile].iterrows():
            data.append([row.tile_id,0,row.xcoord,row.ycoord])

Y = pd.DataFrame(data, columns=['tile_id','cluster','xcoord','ycoord'])
Y.to_csv('swc_zooniverse_cluster_found_coords.csv')

