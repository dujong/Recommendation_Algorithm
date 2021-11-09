import pandas as pd
import numpy as np
from math import log10
import re
from math import log 
from konlpy.tag import Kkma
import os

#by 종두 tfidf 수식
def f(t, d):
    return d.count(t)

def tf(t, d):
    return 0.5 + 0.5*f(t,d)/max([f(w,d) for w in d])

def idf(t, D):
    numerator = len(D)
    denominator = 1 + len([ True for d in D if t in d])
    return log10(numerator/denominator)

def tfidf(t, d, D):
    return tf(t,d)*idf(t, D)

def tokenizer(d):
    
    return d.split()

def tfidfScorer(D):
    D = [re.sub('[^A-Za-z0-9가-힣]', ' ', s) for s in D]
    tokenized_D = [tokenizer(d) for d in D]
    result = []
    for d in tokenized_D:
        result.append([(t, tfidf(t, d, tokenized_D)) for t in d])
    return result

def data_load(file_path):
    return pd.read_csv(file_path)

# by 종두 Data 불러오기
ratings = data_load('data/ratings_3st.csv')
movies = data_load('data/movies_info_new_genres3.csv')

def preprocessing(movies, ratings):
    movies = movies[movies['year'] > 2015]
    movie_ratings = pd.merge(ratings, movies, on='title')
    movie_ratings.drop_duplicates(subset=None, keep='first', inplace=True, ignore_index=False)
    movie_ratings_pivot = movie_ratings.pivot_table(values='rating', index='title', columns='user')
    movie_ratings_pivot = movie_ratings_pivot.fillna(0)

    return movie_ratings, movie_ratings_pivot, movie_ratings_pivot.T

# by 종두 데이터 전처리
movie_ratings, movie_ratings_pivot, movie_ratings_pivot_T = preprocessing(movies, ratings)

# by 종두 pearson 유사도 추출
def pearson_sim(df):
    return df.corr(method='pearson')

pearson_sim = pearson_sim(movie_ratings_pivot)

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import seaborn as sns

# by 종두 K-means로 유사 사용자 추출
def k_means(df):
    # K-Means
    tsne = TSNE(n_components=2, perplexity=10, learning_rate=100).fit_transform(movie_ratings_pivot_T)
    print(tsne.shape)

    name = movie_ratings.columns
    x = tsne[:,0]
    y = tsne[:,1]

    KM = pd.DataFrame()
    KM['x'] = x
    KM['y'] = y
    KM.index = name

    kmeans = KMeans(n_clusters=7)
    kmeans.fit(KM)

    result_by_sklearn = KM.copy()
    result_by_sklearn["cluster"] = kmeans.labels_
    sns.scatterplot(x="x", y="y", hue="cluster", data=result_by_sklearn, palette="Set2");

    plt.show()

    return result_by_sklearn


# by 종두 key : user, value : user의 평균 rating을 넘는 movies
user_movie_dict = dict()

# by 종두 key : movie, value : movie의 Top keyword
keyword = dict()

# by 종두 user들의 평균 rating을 넘는 영화들을 추출하는 함수
def interesting_Movie(movie_ratings_pivot_T, user_movie_dict):
    for i in movie_ratings_pivot_T.index:
        movie_list = []

        avg = np.mean(movie_ratings_pivot_T.loc[i,:])
        li = movie_ratings_pivot_T.loc[i,:] > avg
        for j in range(len(li)):
            if li[j]:
                movie_list.append(li.index[j])

        user_movie_dict[i] = movie_list

    return user_movie_dict


user_movie_dict = interesting_Movie(movie_ratings_pivot_T, user_movie_dict)

# by 종두 keyword 추출
def get_keyword_df(hos, keyword):
    for i, doc in enumerate(tfidfScorer(hos['contents'])):
        print('document:{}'.format(i))
        doc = pd.DataFrame(doc, columns=['word', 'score'])
        doc = keyword_extraction(doc)
        ret = dict((word,score) for word, score in zip(doc.word, doc.score))
        title = hos.loc[i, 'title']
        keyword[title] = ret
    return keyword

def keyword_extraction(doc):
    kkma = Kkma()
    cnt = 0
    for i in doc['word']:
        try:
            NN = kkma.nouns(i)[0]
            if len(str(NN)) < 2:
                continue
            doc.loc[cnt, 'word'] = NN
        except IndexError:
            doc.drop(index=cnt, axis=0, inplace=True)
        cnt += 1
    doc = doc.reset_index(drop=True);
    doc = doc.sort_values(by='score', ascending=False)[:5]
    return doc

movies = movies.reset_index(drop=True)
keyword = pd.DataFrame(get_keyword_df(movies[:10], keyword))
keyword.fillna(0, inplace=True)
similarity_movie = keyword.corr(method='pearson')
similarity_movie

# by 종두 콘텐츠 기반 추천
def recommend_movie_contents(recom_user_name, similarity_movie):
    interest_movie = user_movie_dict[recom_user_name]
    recomm_movie_result = []

    for i in interest_movie:
        recomm_movie_result.append(similarity_movie[i].sort_values(ascending=False).index[1])
    
    return recomm_movie_result

# by 종두 유사 사용자 추출
def sim_user_list(recom_user_name, pear_sim):
    sim_user = pd.DataFrame()
    sim_user['user'] = pear_sim[recom_user_name].index #sorted(pear_sim['CHAEYOOE'], reverse=True)
    sim_user['rating'] = pear_sim[recom_user_name].values
    sim_user = sim_user.sort_values(by='rating', ascending=False)[1:11].reset_index(drop=True)
    return sim_user

# by 종두 협업필터링 예측 평점
def collabor_pred_rating(recom_user_name, sim_user , sim_user_list, recomm_movie_result, movie_ratings_pivot):
          
    pred_ratings = dict()
    #avg_rating = 사용자 i의 평균 평점
    avg_rating = average_ratings(recom_user_name, movie_ratings_pivot)
    
    for i in recomm_movie_result:
        #middle_rating = 사용자 보정 평점
        try:
            sigma_value = 0
            sigma_value2 = 0
            for j in sim_user_list['user']:
                if movie_ratings_pivot.loc[i, j] == 0:
                    continue

                sim_uv = sim_user.loc[recom_user_name,j]
                sigma_value = sigma_value + (sim_uv * (movie_ratings_pivot.loc[i,j] - average_ratings(j, movie_ratings_pivot)))
                sigma_value2 = sigma_value2 + np.abs(sim_uv)
            pred_ratings[i] = avg_rating + (sigma_value / sigma_value2)
        except KeyError:
            continue
    
    return pd.DataFrame([pred_ratings])

# by 종두 협업필터링 기반 추천
def recommend_movie_collabor(recom_user_name, sim_user, similarity_movie, movie_ratings_pivot):
    recomm_movie_result = []

    sim_user_li = sim_user_list(recom_user_name, sim_user)
    for i in sim_user_li['user']:
        #print(index,'번째:',user_movie_dict[i])
        movie_list = user_movie_dict[i]
        for j in movie_list:
            first = similarity_movie[j].sort_values(ascending=False).index[1]
            recomm_movie_result.append(first)
    
    pred_rating = collabor_pred_rating(recom_user_name, sim_user, sim_user_li, recomm_movie_result,movie_ratings_pivot)
    return pred_rating
    # DF 만들기 정렬 후 TOP10개를 추출

    # return recomm_movie_result

recom_user_name = input() #'CHAEYOOE'

sim_user = movie_ratings_pivot.corr(method='pearson')
result_collabor_movie = recommend_movie_collabor(recom_user_name, sim_user, similarity_movie, movie_ratings_pivot)
collaborative_result = result_collabor_movie.T

movie = collaborative_result.sort_values(by = 0, ascending=False)[:10].index
score = collaborative_result.sort_values(by = 0, ascending=False)[:10].values
result = collaborative_result.sort_values(by = 0, ascending=False)[:10]
print(result)


# # by 종두 협업필터링 기반 추천
# def recommend_movie_collabor(recom_user_name, pear_sim, similarity):
#     recomm_movie_result = []

#     sim_user = sim_user_list(recom_user_name, pear_sim)

#     for i in sim_user['user']:
#         #print(index,'번째:',user_movie_dict[i])
#         movie_list = user_movie_dict[i]
#         for j in movie_list:
#             first = similarity[j].sort_values(ascending=False).index[1]
#             recomm_movie_result.append(first)
            
#     return recomm_movie_result




# by 종두 특정 user의 평균 rating
def average_ratings(recom_user_name, movie_ratings_pivot):
    average = movie_ratings_pivot[movie_ratings_pivot[recom_user_name] > 0][recom_user_name]
    average = sum(average) / len(average)
    return average

# 콘텐츠 기반 예측 평점
def contents_pred_rating(recom_user_name, movie_ratings_pivot, movie_ratings_pivot_T):
    avg_rating = average_ratings(recom_user_name, movie_ratings_pivot)
    
    pred_m = pd.DataFrame(index=[recom_user_name])  #추천대상이 되는 사용자를 행으로하는 데이터 프레임 만들기

    for j in range(10):  #len(user['title'].drop_duplicates()) 우선 행개수 10개  
        test=movie_ratings[movie_ratings['title']==movie_ratings['title'].drop_duplicates().iloc[j]] #예측평점을 만들고자 하는 영화를 차례대로 넣기

        sum=0
        for i in range(len(test)):
            test2=movie_ratings[movie_ratings['user']==test.iloc[i]['user']] #특정영화를 본 사용자가 본 전체 영화 저장 


            test3=test2[test2['new_genres']==test.iloc[i]['new_genres']]  #특정사용자가 본 전체 영화중에 특정 영화장르인 모든 영화
            test3['rating'].mean() - test.iloc[i]['rating']  # 특정영화를 본 사용자가 특정영화의 장르에 준 평균평점 - 특정영화에 준 평점

            sum = sum + (test.iloc[i]['rating'] - test3['rating'].mean())

        sum = sum/len(test)
        #print(user['title'].drop_duplicates().iloc[j] , moviesaverage[0] + sum)  # 추천대상이 되는 사용자가 전체영화에 준 평균평점 + (추천영화를 본사람들이 추천 영화장르에 준 평균평점 - 추천영화에 준 평점)
        pred_m[movie_ratings['title'].drop_duplicates().iloc[j]]=avg_rating + sum  #데이터 프레임 열에 영화넣고 예측평점을 값으로 저장 

    return pred_m


# by 종두 협업필터링 예측 평점
def collabor_pred_rating(recom_user_name, similarity_movie, pear_sim ,movie_ratings_pivot, movie_ratings_pivot_T):
    sim_user = sim_user_list(recom_user_name, pear_sim)
    pred_ratings = dict()
    #avg_rating = 사용자 i의 평균 평점
    avg_rating = average_ratings(recom_user_name, movie_ratings_pivot)
    
    for i in movie_ratings_pivot.index:
        #middle_rating = 사용자 보정 평점
        sigma_value = 0
        for j in sim_user:
            if movie_ratings_pivot.loc[i, j] > 0:
                continue

            sim_uv = pear_sim.loc[recom_user_name,j]
            sigma_value = sigma_value + (sim_uv * (movie_ratings_pivot.loc[i,j] - average_ratings(j, movie_ratings_pivot))) / np.abs(sim_uv)
        pred_ratings[i] = avg_rating + sigma_value
    
    return pd.DataFrame([pred_ratings])