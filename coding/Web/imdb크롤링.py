from urllib.request import urlopen
from urllib.request import HTTPError
from bs4 import BeautifulSoup
import pandas as pd

url = pd.read_csv('tttttttttttt.csv')['imdbId']
title_li = []
true_ = 0
false_ = 0

for i in url:
    try:
        html = urlopen(i)
        soup = BeautifulSoup(html, "html.parser")
        title = soup.find('h1', 'TitleHeader__TitleText-sc-1wu6n3d-0 cLNRlG')
        title_li.append(title)
        print('success!')
        true_ += 1

    except HTTPError as e:
        print('page not found')
        title_li.append('page not found')
        false_ += 1

print('True:',true_)
print('False:', false_)