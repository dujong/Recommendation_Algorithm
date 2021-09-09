from selenium.webdriver.chrome.options import Options
from selenium import webdriver
from selenium.common.exceptions import NoSuchElementException
from urllib.request import urlopen
from urllib.request import HTTPError
from bs4 import BeautifulSoup
import pandas as pd
import time
import os

movies_info = pd.read_csv('movies_info.csv')
movie_title = movies_info['title']

new_genres = []
print(movie_title)


driver = webdriver.Chrome(r'chromedriver\chromedriver.exe')
for i in movie_title:

    url = 'https://search.naver.com/search.naver?query=' + str(i)
    driver.get(url)
    html = driver.page_source
    soup = BeautifulSoup(html, 'html.parser')

    try:
        gen = str(soup.select('dl > div > dd')).split('<dd>')[1].split('<span')[0]
        print(gen)
        new_genres.append(gen)
    except IndexError:
        new_genres.append('정보없음')


movies_info['new_genres'] = pd.Series(new_genres)
movies_info.to_csv('movies_info_new_genres.csv', index=False, encoding='utf-8-sig')