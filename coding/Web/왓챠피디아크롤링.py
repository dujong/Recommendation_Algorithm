from selenium.webdriver.chrome.options import Options
from selenium import webdriver
from selenium.common.exceptions import NoSuchElementException
from urllib.request import urlopen
from urllib.request import HTTPError
from bs4 import BeautifulSoup
import pandas as pd
import time
import os


# User 한명
driver = webdriver.Chrome(r'chromedriver\chromedriver.exe')
url = 'https://pedia.watcha.com/ko-KR/users/4WLxZjz7ExroA/contents/movies/ratings'
driver.get(url)

# 영화 제목, link 수집
user = pd.DataFrame(columns=['links'])

for i in range(200):
    driver.execute_script("window.scrollTo(0,document.body.scrollHeight)")
    time.sleep(2)

html = driver.page_source
soup = BeautifulSoup(html, 'html.parser')


movie_href = []
cnt =0

for tag in soup.select('div > ul > li > a'):
    user.loc[cnt, 'links'] = 'https://pedia.watcha.com' + tag['href']+ '/comments'
    movie_href.append('https://pedia.watcha.com' + tag['href'] + '/overview')
    cnt += 1

user.drop(index=0, inplace=True)
user.to_csv('Data/영화_링크.csv', encoding='utf-8-sig')

movie_href.pop(0)
movie_info = pd.DataFrame(columns=['title', 'year', 'country', 'genres', 'time'])
cnt = 0

for url in movie_href:
    html = urlopen(url)
    soup = BeautifulSoup(html, 'html.parser')

    sample = []
    for con in soup.find_all('dd', 'css-11yx0y9-DescriptionDetail e1kvv3953'):
        sample.append(con.text)

        if len(sample) == 5:
            break
    print(sample)

    if len(sample) == 5:
        movie_info.loc[cnt] = sample
    #movie_info = movie_info.append(pd.Series(movie_info_li, index=movie_info.columns), ignore_index=True)
    cnt += 1

movie_info.to_csv('Data/영화정보.csv', encoding='utf-8-sig', index=False)


# 영화 평가한 user, rating 정보 수집
movie_cnt = 0
li = user['links'].tolist()
movie_df = pd.DataFrame(columns=['movie', 'user', 'rating', 'links'])

for mo, li in zip(movie_info['title'], user['links']):
    driver = webdriver.Chrome(r'chromedriver\chromedriver.exe')
    driver.get(li)

    for i in range(100):
        driver.execute_script("window.scrollTo(0,document.body.scrollHeight)")
        time.sleep(2)
    try:
        stores = driver.find_element_by_tag_name('div').text
        stores = stores.splitlines()

        cnt = 0
        for i in range(len(stores)):
            try:
                if stores[i] == '좋아요' and float(stores[i+2]) < 10000:
                    movie_df.loc[cnt] = [mo, stores[i+1], stores[i+2], li]
                    i += 2
            except IndexError:
                continue

            except ValueError:
                continue

            except KeyError:
                continue
    except NoSuchElementException:
        pass

        cnt += 1
    movie_cnt += 1
    movie_df.reset_index(inplace=True)
    movie_df.drop('index', axis=1, inplace=True)

    if not os.path.exists('Data/movies.csv'):
        movie_df.to_csv('Data/movies.csv', encoding='utf-8-sig', mode='w', index=False)
    else:
        movie_df.to_csv('Data/movies.csv', encoding='utf-8-sig', mode='a', header=False, index=False)

