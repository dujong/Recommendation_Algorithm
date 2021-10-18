import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

ratings = pd.read_csv('./movieDatasets/100K/ratings.csv')
movies = pd.read_csv('./movieDatasets/100K/movies.csv')
movie_ratings = pd.merge(ratings, movies, on='movieId')

rating_matrix = movie_ratings.pivot_table(values='rating', index='userId', columns='movieId')
rating_matrix_T = rating_matrix.fillna(0)
rating_matrix = rating_matrix_T.transpose()

df_index = movies['movieId']

df = pd.DataFrame(index=df_index, columns=['timestamp'])

user_sim = rating_matrix.corr(method='pearson')

# sim_u = user_sim[1].sort_values(ascending=False)[1:11].index.tolist()
# sim_r = user_sim[1].sort_values(ascending=False)[1:11].tolist()

def predict_rating(rating_matrix , user__sim):
    ratings_pred = np.dot(rating_matrix, user__sim) / np.array([np.abs(user__sim).sum(axis=1)])
    return pd.DataFrame(data=ratings_pred, index=rating_matrix_T.columns, columns=user_sim.index).transpose()

def get_mae(pred, actual):
    pred = pred[actual.nonzero()].flatten()
    actual = actual[actual.nonzero()].flatten()
    return mean_absolute_error(pred, actual)

def get_mse(pred, actual):
    pred = pred[actual.nonzero()].flatten()
    actual = actual[actual.nonzero()].flatten()
    return mean_squared_error(pred, actual)

pred_matrix = predict_rating(rating_matrix ,user_sim)
result_mae = get_mae(pred_matrix.values, rating_matrix_T.values)
result_mse = get_mse(pred_matrix.values, rating_matrix_T.values)
print("MAE:", result_mae)
print("MSE:", result_mse)

# pred_matrix = np.dot(user_sim,rating_matrix_T) / np.array([np.abs(rating_matrix_T).sum(axis=1)])
# pred_matrix = pd.DataFrame(data=pred_matrix, index=user_sim.index, columns=rating_matrix_T.columns)


# result = []
# cnt = 0
# for j in rating_matrix_T.columns:
#     a = 0
#     for k in sim_u:
#         a += (rating_matrix_T.loc[k][j] * user_sim.loc[1][k])

#     cnt += 1
#     print('cnt :',cnt, 'mid value:', a)
#     result.append(a / len(sim_u))

# df[1] = pd.Series(result)


# print('길이!!!!!!!', len(rating_matrix_T.columns))
# print(len(df))
# print(len(df_index))

# from sklearn.metrics import mean_absolute_error

# def get_mse(pred, actual):
#     pred = pred[actual.nonzero()].flatten()
#     actual = actual[actual.nonzero()].flatten()
#     return mean_absolute_error(pred, actual)

# for i in range(1, 671):
#     sim_u = user_sim[i].sort_values(ascending=False)[1:11].index.tolist()
#     sim_r = user_sim[i].sort_values(ascending=False)[1:11].tolist()

#     result = []
#     for j in rating_matrix_T.columns:
#         a = 0
#         for k in sim_u:
#             a += (rating_matrix_T.loc[k][j] * user_sim.loc[i][k])
#         result.append(a / len(sim_u))
#     df[i] = pd.Series(result)

# df.drop('timestamp', axis=1, inplace=True)
# print(df)