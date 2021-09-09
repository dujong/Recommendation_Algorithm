from selenium.webdriver.chrome.options import Options
from selenium import webdriver
from selenium.common.exceptions import NoSuchElementException
from urllib.request import urlopen
from urllib.request import HTTPError
from bs4 import BeautifulSoup
import pandas as pd
import time
import os

user_info = pd.read_csv('Data/Data0/user_info.csv')

options = webdriver.ChromeOptions()
options.add_experimental_option("excludeSwitches", ["enable-logging"])
driver = webdriver.Chrome(r'chromedriver\chromedriver.exe', options=options)

for i in range(0, 101):
    url = user_info.loc[i, 'href']
    driver.get(url)

    for j in range(150):
        driver.execute_script("window.scrollTo(0,document.body.scrollHeight)")
        time.sleep(2)

    user_rating = pd.DataFrame()
    
    title = []
    rating = []

    cnt = 1
    while True:
        try:
            tit = driver.find_element_by_xpath('//*[@id="root"]/div/div[1]/section/section/div[1]/section/div[1]/div/ul/li[{}]/a/div[2]/div[1]'.format(cnt))
            rat = driver.find_element_by_xpath('//*[@id="root"]/div/div[1]/section/section/div[1]/section/div[1]/div/ul/li[{}]/a/div[2]/div[2]'.format(cnt))
            title.append(tit.text)
            rating.append(str(rat.text).split('★ ')[1])
            cnt += 1

        except IndexError:
            cnt += 1
            continue

        except NoSuchElementException:
            print(cnt)
            break
    user = [user_info.loc[i, 'user']] * len(title)

    user_rating['user'] = pd.Series(user)
    user_rating['title'] = pd.Series(title)
    user_rating['rating'] = pd.Series(rating)

    if not os.path.exists('Data/Data0/ratings.csv'):
        user_rating.to_csv('Data/Data0/ratings.csv', encoding='utf-8-sig', mode='w', index=False)
    else:
        user_rating.to_csv('Data/Data0/ratings.csv', encoding='utf-8-sig', mode='a', header=False, index=False)

