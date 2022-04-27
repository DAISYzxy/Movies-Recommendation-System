from flask import Flask, url_for, request, render_template
import pandas as pd
import numpy as np

from service.NGCF import Recommendation_NGCF
from service.DSSM import Recommendation_DSSM
from service.data import initial_data

from service.svd import recommendation_svd

app = Flask(__name__)
app.secret_key = "super secret key"

def recommendation_direct(age, gender, occupation):
    # recommendation method should be added!
    df_recommendation = pd.DataFrame({'movieId': np.array([1] * 10, dtype='int32'),
                                      'title': 'Toy Story',
                                      'poster_url': '',
                                   #    'poster_url': 'https://images-na.ssl-images-amazon.com/images/M/MV5BMDU2ZWJlMjktMTRhMy00ZTA5LWEzNDgtYmNmZTEwZTViZWJkXkEyXkFqcGdeQXVyNDQ2OTk4MzI@..jpg',
                                      'score': 3.5
                                      })

    recommendation = list(zip(list(df_recommendation.title),
                              list(df_recommendation.poster_url),
                              list(df_recommendation.score)))
    # print(recommendation)

    return recommendation

def movie_ranking(age, gender, occupation):
    # ranking method should be added!
    df_recommendation = pd.DataFrame({'movieId': np.array([1] * 10, dtype='int32'),
                                      'title': 'Toy Story',
                                      'poster_url': 'https://images-na.ssl-images-amazon.com/images/M/MV5BMDU2ZWJlMjktMTRhMy00ZTA5LWEzNDgtYmNmZTEwZTViZWJkXkEyXkFqcGdeQXVyNDQ2OTk4MzI@..jpg',
                                      })

    moviestorank = list(zip(list(df_recommendation.title),
                            list(df_recommendation.poster_url)))

    return moviestorank

def recommendation_with_rank(age, gender, occupation, movie_0, movie_1, movie_2, movie_3, movie_4, movie_5,
                             movie_6,
                             movie_7, movie_8, movie_9):
    # recommendation method with both user info and ranking info should be added!
    df_recommendation = pd.DataFrame({'movieId': np.array([1] * 10, dtype='int32'),
                                      'title': 'Toy Story',
                                      'poster_url': 'https://images-na.ssl-images-amazon.com/images/M/MV5BMDU2ZWJlMjktMTRhMy00ZTA5LWEzNDgtYmNmZTEwZTViZWJkXkEyXkFqcGdeQXVyNDQ2OTk4MzI@..jpg',
                                      'score': 3.5
                                      })

    recommendation = list(zip(list(df_recommendation.title),
                              list(df_recommendation.poster_url),
                              list(df_recommendation.score)))

    return recommendation


df_ML_movies,df_users,df_movies,df_ratings,df_posters = initial_data()
# movieid_list = [296,1225,1288,1027,1343,10,380,1320,1030,356]

@app.route('/recommend', methods=["GET", "POST"])
def user_info():
    if request.method == 'POST':
        age = request.form['age']
        gender = request.form['gender']
        occupation = request.form['occupation']

        # DSSM模型
        if 'modelA' in request.form:
            root_dir = '/Users/asteriachiang/Documents/5001_Foundations_of_Data_Analytics/model/'
            recommendation_list, scores_list = Recommendation_DSSM('F', 18, occupation, 10, root_dir)
            global movieid_list
            movieid_list = recommendation_list
            recommendation = list(zip(list(df_ML_movies[df_ML_movies.MovieID.isin(recommendation_list)].Title.unique()),
                                      list(df_ML_movies[
                                               df_ML_movies.MovieID.isin(recommendation_list)].PosterUrl.unique())
                                      ))
            return render_template('movie.html', recommendation=recommendation)

        # 需要rank信息，默认使用DSSM预推荐10部电影
        root_dir = '/Users/asteriachiang/Documents/5001_Foundations_of_Data_Analytics/model/'
        recommendation_list, scores_list = Recommendation_DSSM('F', 18, occupation, 10, root_dir)
        movieid_list=recommendation_list
        print(movieid_list)
        recommendation = list(zip(list(df_ML_movies[df_ML_movies.MovieID.isin(recommendation_list)].Title.unique()),
                                  list(df_ML_movies[df_ML_movies.MovieID.isin(recommendation_list)].PosterUrl.unique())
                                  ))

        return render_template('ranking.html', moviestorank=recommendation,age=age,gender=gender,occupation=occupation)

    else:
        age = request.args.get('age')
        gender = request.args.get('gender')
        occupation = request.args.get('occupation')
        if 'modelA' in request.args:
            recommendation = recommendation_direct(age, gender, occupation)
            return render_template('movie.html', recommendation=recommendation)

        moviestorank = movie_ranking(age, gender, occupation)
        return render_template('ranking.html', moviestorank=moviestorank,age=age,gender=gender,occupation=occupation)


@app.route('/rank', methods=["GET", "POST"])
def rank():
    if request.method == 'POST':
        movie_0 = request.form['movie-0']
        movie_1 = request.form['movie-1']
        movie_2 = request.form['movie-2']
        movie_3 = request.form['movie-3']
        movie_4 = request.form['movie-4']
        movie_5 = request.form['movie-5']
        movie_6 = request.form['movie-6']
        movie_7 = request.form['movie-7']
        movie_8 = request.form['movie-8']
        movie_9 = request.form['movie-9']
        age = request.form['age']
        gender = request.form['gender']
        occupation = request.form['occupation']

        ratingArr = [int(movie_0), int(movie_1), int(movie_2), int(movie_3), int(movie_4),
                     int(movie_5), int(movie_6), int(movie_7), int(movie_8), int(movie_9)]

        if 'modelC' in request.form:
            recommendation_list,scores_list = recommendation_svd(ratingArr, movieid_list, 10, df_ML_movies)
            recommendation = list(zip(list(df_ML_movies[df_ML_movies.MovieID.isin(recommendation_list)].Title.unique()),
                                  list(df_ML_movies[df_ML_movies.MovieID.isin(recommendation_list)].PosterUrl.unique()),
                                  scores_list))

            return render_template('movie.html', recommendation=recommendation)

        if 'modelD' in request.form:
            print(movieid_list)
            root_dir = '/Users/asteriachiang/Documents/5001_Foundations_of_Data_Analytics/model/'
            recommendation_list,scores_list = Recommendation_NGCF(ratingArr, movieid_list, 10, root_dir)
            recommendation = list(zip(list(df_ML_movies[df_ML_movies.MovieID.isin(recommendation_list)].Title.unique()),
                                      list(df_ML_movies[df_ML_movies.MovieID.isin(recommendation_list)].PosterUrl.unique()),
                                      scores_list))

            return render_template('movie.html', recommendation=recommendation)

    else:
        movie_0 = request.args.get('movie-0')
        movie_1 = request.args.get('movie-1')
        movie_2 = request.args.get('movie-2')
        movie_3 = request.args.get('movie-3')
        movie_4 = request.args.get('movie-4')
        movie_5 = request.args.get('movie-5')
        movie_6 = request.args.get('movie-6')
        movie_7 = request.args.get('movie-7')
        movie_8 = request.args.get('movie-8')
        movie_9 = request.args.get('movie-9')
        age = request.args.get('age')
        gender = request.args.get('gender')
        occupation = request.args.get('occupation')


        if 'modelC' in request.args:
            ratingArr = [movie_0, movie_1, movie_2, movie_3, movie_4,
                     movie_5, movie_6, movie_7, movie_8, movie_9]
            movieids = [1230,2664,2019,3201,1921,642,1193,402,872,989]
            recommendation_list = recommendation_svd(ratingArr, movieids, 10, df_ML_movies)
            recommendation = list(zip(list(df_ML_movies[df_ML_movies.MovieID.isin(recommendation_list)].Title.unique()),
                                  list(df_ML_movies[df_ML_movies.MovieID.isin(recommendation_list)].PosterUrl.unique()),
                                  [1,2,3,4,5,6,7,8,9,10]))
            
            return render_template('movie.html', recommendation=recommendation)

if __name__ == '__main__':
    app.run(debug=True)
