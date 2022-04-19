# -*- coding: utf-8 -*-
"""
Created on Tue Apr 19 18:08:05 2022

@author: MXR
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn import metrics
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder

from tensorflow.keras.layers import Embedding, Reshape, Concatenate, Dot, Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import Callback, EarlyStopping, ModelCheckpoint

from tensorflow.keras.models import load_model
import tensorflow.keras as keras
import tensorflow.keras.backend as K
import tensorflow as tf

import warnings
warnings.filterwarnings('ignore')

# root_dir = "E:/Study/HKUST/2-Foundation of Data Analytics/GroupProject/"
root_dir = '' # please set the root_dir to the folder which stores the model and the file
#%%
# an example on parameters: Recommendation_DSSM('F',18, 20, 10, root_dir)
def Recommendation_DSSM(gender, age, occupation, top_N, root_dir):
    df = pd.read_csv(root_dir + "MovieLens_IMDB.csv", index_col = 0)
    df_sample = df.drop_duplicates(subset = ['MovieID'])#sample(100000, replace = False)
    MovieID_sample = np.array(df_sample['MovieID']).reshape(-1,)
    users = pd.read_csv(root_dir + 'ml-1m/users.dat', sep='::', header=None, 
                     names=['UserID', 'Gender', 'Age', 'Occupation', 'Zip-code'])

    users.loc['pred_user'] = [0, gender, age, occupation, 'X']

    le = LabelEncoder()
    users['Gender'] = le.fit_transform(users['Gender'])
    
    # one-hot encoding for occupation
    occ_onehot = pd.get_dummies(users['Occupation'])
    users = users.drop(columns='Occupation')
    users = users.join(occ_onehot)
    users = users[['Gender','Age',0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]]
    users = users.astype(np.float32)
    users = (users - users.min(axis = 0))/(users.max(axis = 0) - users.min(axis = 0))
    user_feat = users.loc[['pred_user']]

    X_user = user_feat.sample(frac = len(df_sample), replace = True)
    X_movie = df_sample[['startYear', 'Fantasy', 'Comedy', 'Western', \
                  'Adventure', 'Mystery', 'Animation', 'Thriller', "Children's", \
                  'Romance', 'Crime', 'War', 'Drama', 'Action', 'Sci-Fi', 'Musical', \
                  'Horror', 'Film-Noir', 'Documentary', 'averageRating', 'numVotes',\
                  ]]
    
    X_movie = (X_movie - X_movie.min(axis = 0))/(X_movie.max(axis = 0) - X_movie.min(axis = 0))
    
    model_pretrain = load_model(root_dir + "DSSM.h5")
    
    pred = model_pretrain.predict([X_user, X_movie])
    pred = np.array(pred).reshape(-1,)
    tmp = dict(zip(list(pred),list(MovieID_sample)))
    tmp= sorted(tmp.items(),reverse=True)
    # return [x[1] for x in tmp][0:top_N]
    return [x[1] for x in tmp][0:top_N], [x[0] for x in tmp][0:20]

#%%
root_dir = "E:/Study/HKUST/2-Foundation of Data Analytics/GroupProject/"  ## change the directory to your path
print(Recommendation_DSSM('M',65, 6, 20, root_dir))

    