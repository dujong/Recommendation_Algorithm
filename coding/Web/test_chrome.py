import pandas as pd
import numpy as np

Data1_movies = pd.read_csv('./Data/Data1/movies.csv',encoding='utf-8-sig')
Data1_movie_info = pd.read_csv('./Data/Data1/영화정보.csv',encoding='utf-8-sig')

Data2_movies = pd.read_csv('./Data/Data2/movies.csv',encoding='utf-8-sig')
Data2_movie_info = pd.read_csv('./Data/Data2/영화정보.csv',encoding='utf-8-sig')

Data3_movies = pd.read_csv('./Data/Data3/movies.csv',encoding='utf-8-sig')
Data3_movie_info = pd.read_csv('./Data/Data3/영화정보.csv',encoding='utf-8-sig')

Data4_movies = pd.read_csv('./Data/Data4/movies.csv',encoding='utf-8-sig')
Data4_movie_info = pd.read_csv('./Data/Data4/영화정보.csv',encoding='utf-8-sig')

Data5_movies = pd.read_csv('./Data/Data5/movies.csv',encoding='utf-8-sig')
Data5_movie_info = pd.read_csv('./Data/Data5/영화정보.csv',encoding='utf-8-sig')

Data6_movies = pd.read_csv('./Data/Data6/movies.csv',encoding='utf-8-sig')
Data6_movie_info = pd.read_csv('./Data/Data6/영화정보.csv',encoding='utf-8-sig')


movies = pd.concat([Data1_movies, Data2_movies, Data3_movies, Data4_movies, Data5_movies, Data6_movies])
print('movies :', len(movies))
movies = movies.drop_duplicates(subset=None, keep='first', inplace=False, ignore_index=False)
print('movies 중복제거 후:', len(movies))

movies_info = pd.concat([Data1_movie_info, Data2_movie_info, Data3_movie_info, Data4_movie_info, Data5_movie_info, Data6_movie_info])
print('movies_info :', len(movies_info))
movies_info = movies_info.drop_duplicates(['title'], keep = 'first')
print('movies_info :', len(movies_info))

movies.to_csv('movies.csv', index=False, encoding='utf-8-sig')
movies_info.to_csv('movies_info.csv', index=False, encoding='utf-8-sig')

