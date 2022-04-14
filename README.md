# Movie-Recommendation-System

> A movie recommendation system uses MovieLens dataset and IMDb dataset.

## Data Exploration Reports
Bellow we have a profiler report of the 3 dataframes. They represent the result of merging the Movielens and IMDb datasets:

- [Movies](https://htmlpreview.github.io/?https://github.com/Movinder/movielens-imdb-exploration/blob/master/data/reports/movies_report.html)
- [Ratings](https://htmlpreview.github.io/?https://github.com/Movinder/movielens-imdb-exploration/blob/master/data/reports/ratings_report.html)
- [Users](https://htmlpreview.github.io/?https://github.com/Movinder/movielens-imdb-exploration/blob/master/data/reports/users_report.html)



### Required packages
The code has been tested running under Python 3.6.5, with the following packages installed (along with their dependencies):
- tensorflow-gpu == 1.4.0
- numpy == 1.14.5
- sklearn == 0.19.1


### Running the code
```
$ cd src
$ python preprocess.py --dataset movie (or --dataset book)
$ python main.py --dataset movie (note: use -h to check optional arguments)
```
