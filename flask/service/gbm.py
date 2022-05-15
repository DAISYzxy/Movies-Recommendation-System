import pandas as pd
import numpy as np
from sklearn import metrics
import os
import itertools
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
import heapq
import warnings
warnings.filterwarnings('ignore')

# import lightgbm as lgb
from lightgbm import LGBMRegressor

def train_test_preprocessing(new_user, df, df_movie):
    '''
    new_user: [Gender, Age, Occupation]
    df: info of user & movie
    df_movie: info of movie (with url)
    '''

    gender_list = ['F', 'M']
    age_list = ["1", "18", "25", "35", "45", "50", "56"]
    occ_list = ["other", "academic/educator", "artist", "clerical/admin", "college/grad student",
                "customer service", "doctor/health care", "executive/managerial", "farmer",
                "homemaker", "K-12 student", "lawyer", "programmer", "retired", "sales/marketing",
                "scientist", "self-employed", "technician/engineer", "tradesman/craftsman",
                "unemployed", "writer"]

    new_user_info = pd.DataFrame(np.zeros([15, 23]))
    new_user_info.columns = ['Gender', 'Age', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
                             '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20']
    new_user_info['Gender'] = gender_list.index(new_user[0])
    new_user_info['Age'] = age_list.index(new_user[1])
    new_user_info[new_user[2]] = 1

    # training set
    x_train = df.drop(columns=['MovieID', 'UserID', 'Rating', 'runtimeMinutes'])
    y_train = df['Rating']

    # test set
    x_test = pd.DataFrame(np.repeat(new_user_info[:1].values, df_movie.shape[0], axis=0),
                          columns=new_user_info.columns)
    x_test = pd.concat([df_movie.drop(columns=['MovieID', 'runtimeMinutes', 'PosterUrl', 'originalTitle']),
                        x_test], axis=1)

    return x_train, y_train, x_test


def Recommendation_gbm(new_user, top_k, df, df_movie):
    '''
    top_k: number of movies recommended
    '''

    x_train, y_train, x_test = train_test_preprocessing(new_user, df, df_movie)

    sc = StandardScaler()
    x_train = sc.fit_transform(x_train)
    x_test = sc.transform(x_test)

    gbm = LGBMRegressor()
    gbm.fit(x_train, y_train)
    y_pred = gbm.predict(x_test).tolist()

    rec_id = []
    rec_score = heapq.nlargest(top_k, y_pred)
    title_list = []
    url_list = []

    for score in rec_score:
        index = y_pred.index(score)
        rec_id.append(df_movie.iloc[index, 0])
        title_list.append(df_movie[df_movie.MovieID.isin([df_movie.iloc[index, 0]])].originalTitle.to_numpy()[0])
        url_list.append(df_movie[df_movie.MovieID.isin([df_movie.iloc[index, 0]])].PosterUrl.to_numpy()[0])

    return rec_id, rec_score, title_list, url_list

# df = pd.read_csv('../data/df.csv')
# df_movie = pd.read_csv('../data/df_movie.csv')
# print(len(df_movie))
# print(len(df_movie))
# df_movie.drop_duplicates(subset=['MovieID'], keep='first', inplace=True)
# df_movie.drop_duplicates(subset=['originalTitle'], keep='first', inplace=True)
# print(len(df_movie))


# # movielens
# df_users = pd.read_csv("../../Data/movielens/users.dat", sep = "::", header=None, engine='python',
#                names=["UserID", "Gender", "Age", "Occupation", "Zip-code"])
# df_movies = pd.read_csv("../../Data/movielens/movies.dat", sep = "::", header=None, engine='python',
#                names=["MovieID", "Title", "Genres"], encoding='ISO-8859-1')
# df_ratings = pd.read_csv("../../Data/movielens/ratings.dat", sep="::", header=None, engine='python',
#                       names=["UserID", "MovieID", "Rating", "Timestamp"])
#
# # urls
# df_posters = pd.read_csv("../data/movie_poster.csv", names=["MovieID", "PosterUrl"])
#
# # ALL
# df = pd.merge(df_movies, df_ratings, on="MovieID")
# df = pd.merge(df, df_users, on="UserID")
# df_ML_movies = pd.merge(df, df_posters, on="MovieID")
#
# list=[1360,1387,910,1384,1154,829,23,189,194,977]
# title_list = []
# url_list = []
# print(list)
# for id in list:
#     if (df_posters[df_posters.MovieID.isin([id])].MovieID.empty):
#         title_list.append(df_movies[df_movies.MovieID.isin([id])].Title.to_numpy()[0])
#         url_list.append('')
#     else:
#         title_list.append(df_ML_movies[df_ML_movies.MovieID.isin([id])].Title.to_numpy()[0])
#         url_list.append(df_ML_movies[df_ML_movies.MovieID.isin([id])].PosterUrl.to_numpy()[0])
#
# print(len(url_list))
# print(url_list)
