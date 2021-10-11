import pandas as pd
import numpy as np
from math import log10
import re
from math import log 
from konlpy.tag import Kkma
import os

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


# dict 형태로 movie - keyword dict 저장해놓기
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
    doc = doc.sort_values(by='score', ascending=False)[:3]
    return doc


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

# by 종두 유사 사용자 추출
movie_ratings, movie_ratings_pivot, movie_ratings_pivot_T = preprocessing(movies, ratings)


def pearson_sim(df):
    return df.corr(method='pearson')

pearson_sim = pearson_sim(movie_ratings_pivot)

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import seaborn as sns
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


# by 종두 
user_movie_dict = dict()
user_movie_keyword = dict()
keyword = dict()


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

movies = movies.reset_index(drop=True)
keyword = get_keyword_df(movies[:10], keyword)

df = pd.DataFrame(keyword)
df.fillna(0, inplace=True)

similarity = pearson_sim(df)

def average_ratings(recom_user_name, movie_ratings_pivot):
    average = movie_ratings_pivot[movie_ratings_pivot[recom_user_name] > 0][recom_user_name]
    average = sum(average) / len(average)
    return average

def contents_pred_rating(movie_ratings_pivot_T):
    CHAEYOOE_average = movie_ratings_pivot[movie_ratings_pivot['CHAEYOOE'] > 0]['CHAEYOOE']
    CHAEYOOE_average = sum(average_ratings) / len(average_ratings)

def collabor_pred_rating(recom_user_name, movie_ratings_pivot, movie_ratings_pivot_T, similarity):
    for i in movie_ratings_pivot_T.index:
        #avg_rating = 사용자 i의 평균 평점
        avg_rating = average_ratings(recom_user_name, movie_ratings_pivot)
        #middle_rating = 사용자 보정 평점
        middle_rating = np.dot(movie_ratings_pivot, similarity)

    #CHAEYOOE_average = movie_ratings_pivot[movie_ratings_pivot['CHAEYOOE'] > 0]['CHAEYOOE']
    #CHAEYOOE_average = sum(average) / len(average)


# recom_user_name = 'CHAEYOOE'

# def similarity_movie(recom_user_name, movie_title, similarity, contents):
#     num = round(len(similarity.index) * 0.2)
#     sim_movie = pd.DataFrame(similarity.loc[movie_title, :]).sort_values(by=movie_title, ascending=False)[1:num]
#     pred_movie = sim_movie.index
#     pred_rating = 0

# chaeyooe 대상으로 했을 때, movie_title은 대상이 흥미있는 영화들이고, 그 영화와 비슷한 영화





# @@@ 다른 사용자들이 특정 영화에 매긴 rating의 평균 점수 code
# avg = []

# for i in first.index:
#     avg.append(np.mean(ratings[ratings['title'] == i]['rating']))

# first['sim_user_rating'] = avg

movie_ratings_pivot_t = movie_ratings_pivot.T
movie_ratings_pivot_t

sim_user = movie_ratings_pivot_t[movie_ratings_pivot_t['유전'] > 0].index
pred = []
for i in sim_user:
    sim = pearson_sim.loc['CHAEYOOE', i]
    sco = movie_ratings_pivot_t.loc[i, '유전']
    aver = average(i)
    pred.append(sim*(sco-aver)/ sim)
middle_result = np.mean(pred)

CHAEYOOE_average = movie_ratings_pivot[movie_ratings_pivot['CHAEYOOE'] > 0]['CHAEYOOE']
CHAEYOOE_average = sum(CHAEYOOE_average) / len(CHAEYOOE_average)

CHAEYOOE_average + middle_result