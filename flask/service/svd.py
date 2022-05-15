import random
import numpy as np
from surprise import Dataset, Reader, NMF
import pandas as pd

'''
  Simulation.
'''


def generate(MovieIDs, num_mvs, df):
    movie_ids = [random.randint(1, len(df.MovieID.unique())) for x in range(num_mvs)]
    rating_list = [round(random.uniform(1, 5)) for x in range(num_mvs)]

    return movie_ids, rating_list


'''
  Recommendation.
'''
def recommendation_svd(ratingArr, movieids, num_movie, df):
    ratings_dict = {'MovieID': list(df.MovieID) + list(movieids),
                    'UserID': list(df.UserID) + [max(df.UserID) + 1 for i in range(num_movie)],
                    'Rating': list(df.Rating) + ratingArr}

    new_df = pd.DataFrame(ratings_dict)
    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(new_df[['UserID', 'MovieID', 'Rating']], reader)

    trainset = data.build_full_trainset()
    userIds = trainset.to_inner_uid(max(new_df.UserID))

    nmf = NMF()
    nmf.fit(trainset)
    mat = np.dot(nmf.pu, nmf.qi.T)

    mv_score = mat[userIds, :]
    best_mv_ids = (-mv_score).argsort()[:num_movie]
    scores = mv_score[best_mv_ids]

    for id, s in enumerate(scores):
        scores[id] = min(5., round(s, 3))

    return best_mv_ids, scores
