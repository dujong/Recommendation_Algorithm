from bs4 import BeautifulSoup
import pandas as pd
from urllib.request import urlopen

url = 'https://pedia.watcha.com/ko-KR/contents/mOAk9JQ/overview'
html = urlopen(url)
soup = BeautifulSoup(html, 'html.parser')

movie_info = pd.DataFrame(columns=['title', 'year', 'country', 'genres', 'time', 'age_rating'])

movie_info_li = []
for con in soup.find_all('dd', 'css-11yx0y9-DescriptionDetail e1kvv3953'):
    movie_info_li.append(con.text)

movie_info.loc[0] = movie_info_li
#movie_info = movie_info.append(pd.Series(movie_info_li, index=movie_info.columns), ignore_index=True)

print(movie_info)