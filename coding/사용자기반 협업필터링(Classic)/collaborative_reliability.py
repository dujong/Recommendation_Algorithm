import pandas as pd
import numpy as np
import random
from pandas.core.frame import DataFrame
from sklearn.metrics.pairwise import cosine_similarity
from numpy.linalg import norm

# csv 불러오기
movies = pd.read_csv('./movieDatasets/movies.csv')
ratings = pd.read_csv('./movieDatasets/ratings.csv')

ratings.drop('timestamp', axis=1, inplace=True)

rating_movies = pd.merge(ratings, movies, on='movieId')
##rating_movies = pd.merge(rating_movies, reliability, on='userId')

# 0.5 수치는 우리가 정한다!!!
##rating_movies = rating_movies[rating_movies['T'] > 0.5]

# 사용자 - 아이템 matrix 생성
ratings_matrix = rating_movies.pivot_table(values='rating', index='userId', columns='title')

# Null 값 0으로 변경
ratings_matrix = ratings_matrix.fillna(0)

def cos_sim(a,b):
    b = b.transpose()
    return np.dot(a,b)/(norm(a)*norm(b))


# cosine_similarity를 사용해서 사용자 유사도 측정
def get_user_sim(ratings_matrix, userId):
    user_sim = cosine_similarity(ratings_matrix, ratings_matrix)
    user_sim = pd.DataFrame(data=user_sim, index=ratings_matrix.index, columns=ratings_matrix.index)
    similar_users = user_sim.loc[userId,:]
    similar_user = similar_users.sort_values(ascending=False)[1:].index[0:5]
    return user_sim, similar_user

user_sim, similar_user = get_user_sim(ratings_matrix, 1)
    
# 사용자가 안본 list 추출
def unseen_movie(ratings_matrix, userId):
    user_rating = ratings_matrix.loc[userId,:]
    movies_list = user_rating[user_rating==0]
    
    return movies_list.index.tolist()

# 콘텐츠 추천
def recomm_movie_by_userId(ratings_matrix, movies_list):
    sim_user_recomm = list()
    recomm_movies = list()
    
    # 비슷한 사용자들이 좋게 평가한 movie list 추출
    for i in similar_user:
        sim_user_recomm.append(ratings_matrix.loc[i,:].sort_values(ascending=False).index.tolist()[:3])
    
    # 2차원 리스트 -> 1차원 리스트
    sim_user_recomm = np.array(sim_user_recomm).flatten().tolist()
    
    # 좋게 평가된 movie list에서 추천 받을 사용자가 보지 않은 list 추출
    for i in sim_user_recomm:
        if i in movies_list:
            recomm_movies.append(i)
    
    # 걸려진 데이터 중에서 random을 뽑아서 실행
    random_number = list()
    
    for i in range(3):
        random_number.append(recomm_movies[random.randint(0, len(recomm_movies)-1)])
        
    # 중복 제거
    return set(random_number)

unseen_movie = unseen_movie(ratings_matrix, 1)
recomm_movies = recomm_movie_by_userId(ratings_matrix, unseen_movie)
print(recomm_movies)