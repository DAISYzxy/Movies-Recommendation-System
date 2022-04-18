from surprise import SVD, NMF
from surprise import Dataset
from surprise.model_selection import train_test_split
from surprise import Reader
import pandas as pd
import numpy as np
from collections import defaultdict

def get_top_n(predictions, n):
    '''
    '从一个预测集中为每个用户返回top-N个推荐'

    参数：
        predictions(list of Prediction objects): 预测对象列表，由某个用于预测的算法返回.
            —————————————————————————————————————————————————————————————————
        n(int):为每个用户进行的推荐的数量。 默认值为10.
            —————————————————————————————————————————————————————————————————

    返回：
        一个字典，字典的键是用户（原始）ID，字典对应的值是为这个用户推荐的n个元组的列表：
        [(物品1原始id, 评分预测1), ...,(物品n原始id, 评分预测n)]
    '''
    top_n = defaultdict(list)
    for uid, iid, true_r, est, _ in predictions:
        top_n[uid].append((iid, est))

    for uid, user_ratings in top_n.items():
        user_ratings.sort(key=lambda x: x[1], reverse=True)
        top_n[uid] = user_ratings[:n]

    return top_n

def recommendation_svd(ratingArr, movieids, num_movie,ratings):

    ratings_dict = {'MovieID': list(ratings.MovieID) + list(movieids),
                  'UserID': list(ratings.UserID) + [max(ratings.UserID+1) for i in range(num_movie)],
                  'Rating': list(ratings.Rating) + ratingArr}

    df = pd.DataFrame(ratings_dict)
    reader = Reader(rating_scale=(1,5))
    data = Dataset.load_from_df(df[['UserID', 'MovieID', 'Rating']], reader)
    print(data)
    trainset, testset = train_test_split(data, test_size=.25)
    svd = SVD()
    svd.fit(trainset)
    predictions = svd.test(testset)
    top_n = get_top_n(predictions, n=num_movie)
    return [ids for (ids, _) in top_n[max(ratings.UserID)]]


