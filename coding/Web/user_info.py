from selenium.webdriver.chrome.options import Options
from selenium import webdriver
from selenium.common.exceptions import NoSuchElementException
from urllib.request import urlopen
from urllib.request import HTTPError
from bs4 import BeautifulSoup
import pandas as pd
import time
import os

users = pd.DataFrame()

driver = webdriver.Chrome(r'chromedriver\chromedriver.exe')
url = 'https://pedia.watcha.com/ko-KR/contents/mWyaxMN/comments'
driver.get(url)

for i in range(100):
    driver.execute_script("window.scrollTo(0,document.body.scrollHeight)")
    time.sleep(2)

html = driver.page_source
soup = BeautifulSoup(html, 'html.parser')

user_list = []
user_href = []
cnt = 1
for i in range(200):
    try:
        a = driver.find_element_by_xpath('//*[@id="root"]/div/div[1]/section/section/div/div/div/ul/div[{}]/div[1]/div[1]/a'.format(cnt))
        user_list.append(a.text)
        user_href.append(str(a.get_attribute('href')) + '/contents/movies/ratings')
        cnt += 1
    except NoSuchElementException:
        continue

users['user'] = pd.Series(user_list)
users['href'] = pd.Series(user_href)


users.to_csv('./Data/Data0/user_info.csv', index=False, encoding='utf-8-sig')

