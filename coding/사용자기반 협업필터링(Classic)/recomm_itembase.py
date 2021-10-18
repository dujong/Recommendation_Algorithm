import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.linear_model import LinearRegression

ratings = pd.read_csv('./movieDatasets/100K/ratings.csv')
movies = pd.read_csv('./movieDatasets/100K/movies.csv')
movie_ratings = pd.merge(ratings, movies, on='movieId')


movie_user_rating = movie_ratings.pivot_table('rating', index='userId', columns='title')
#user_movie_rating = movie_ratings.pivot_table('rating', index='userId', columns='title')
movie_user_rating.fillna(0, inplace=True)

#print(movie_user_rating)
# df_index = movies['movieId']
#tem_based_collabor = cosine_similarity(movie_user_rating)
# df = pd.DataFrame(index=df_index, columns=['timestamp'])

movie_sim = movie_user_rating.corr(method='pearson')
# print(movie_user_rating)

def predict_rating(movie_user_rating, movie_sim):
    ratings_pred = np.dot(movie_user_rating, movie_sim) / np.array([np.abs(movie_sim).sum(axis=1)])
    return pd.DataFrame(data=ratings_pred, index=movie_user_rating.index, columns=movie_sim.columns)

def get_mae(pred, actual):
    pred = pred[actual.nonzero()].flatten()
    actual = actual[actual.nonzero()].flatten()
    return mean_absolute_error(pred, actual)

def get_mse(pred, actual):
    pred = pred[actual.nonzero()].flatten()
    actual = actual[actual.nonzero()].flatten()
    return mean_squared_error(pred, actual)

pred_matrix = predict_rating(movie_user_rating, movie_sim)

result_mae = get_mae(pred_matrix.values, movie_user_rating.values)
result_mse = get_mse(pred_matrix.values, movie_user_rating.values)
print("MAE:", result_mae)
print("MSE:", result_mse)
