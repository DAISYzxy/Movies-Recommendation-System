from flask import Flask, url_for, request, render_template
import pandas as pd
import numpy as np

app = Flask(__name__)


def recommendation_direct(age, gender, occupation, zipcode):
    # recommendation method should be added!
    df_recommendation = pd.DataFrame({'movieId': np.array([1] * 10, dtype='int32'),
                                      'title': 'Toy Story',
                                      'poster_url': 'https://images-na.ssl-images-amazon.com/images/M/MV5BMDU2ZWJlMjktMTRhMy00ZTA5LWEzNDgtYmNmZTEwZTViZWJkXkEyXkFqcGdeQXVyNDQ2OTk4MzI@..jpg',
                                      'score': 3.5
                                      })

    recommendation = list(zip(list(df_recommendation.title),
                              list(df_recommendation.poster_url),
                              list(df_recommendation.score)))
    # print(recommendation)

    return recommendation


def movie_ranking(age, gender, occupation, zipcode):
    # ranking method should be added!
    df_recommendation = pd.DataFrame({'movieId': np.array([1] * 10, dtype='int32'),
                                      'title': 'Toy Story',
                                      'poster_url': 'https://images-na.ssl-images-amazon.com/images/M/MV5BMDU2ZWJlMjktMTRhMy00ZTA5LWEzNDgtYmNmZTEwZTViZWJkXkEyXkFqcGdeQXVyNDQ2OTk4MzI@..jpg',
                                      })

    moviestorank = list(zip(list(df_recommendation.title),
                            list(df_recommendation.poster_url)))

    return moviestorank


def recommendation_with_rank(age, gender, occupation, zipcode, movie_0, movie_1, movie_2, movie_3, movie_4, movie_5,
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


@app.route('/recommend', methods=["GET", "POST"])
def user_info():
    if request.method == 'POST':
        age = request.form['age']
        gender = request.form['gender']
        occupation = request.form['occupation']
        zipcode = request.form['zipcode']
        if 'find_direct' in request.form:
            recommendation = recommendation_direct(age, gender, occupation, zipcode)
            return render_template('movie.html', recommendation=recommendation)

        moviestorank = movie_ranking(age, gender, occupation, zipcode)
        return render_template('ranking.html', moviestorank=moviestorank,age=age,gender=gender,occupation=occupation,zipcode=zipcode)

    else:
        age = request.args.get('age')
        gender = request.args.get('gender')
        occupation = request.args.get('occupation')
        zipcode = request.args.get('zipcode')
        if 'find_direct' in request.args:
            recommendation = recommendation_direct(age, gender, occupation, zipcode)
            return render_template('movie.html', recommendation=recommendation)


        moviestorank = movie_ranking(age, gender, occupation, zipcode)
        return render_template('ranking.html', moviestorank=moviestorank,age=age,gender=gender,occupation=occupation,zipcode=zipcode)


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
        zipcode = request.form['zipcode']


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
        zipcode = request.args.get('zipcode')

    recommendation = recommendation_with_rank(age, gender, occupation, zipcode, movie_0, movie_1, movie_2, movie_3,
                                              movie_4, movie_5, movie_6,
                                              movie_7, movie_8, movie_9)

    return render_template('movie.html', recommendation=recommendation)


if __name__ == '__main__':
    app.run(debug=True)
