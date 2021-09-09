import pandas as pd
from math import log10
import pandas as pd
import re
from math import log 
from konlpy.tag import Kkma
import os

test = pd.read_csv('train.csv', encoding='cp949')

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

kkma = Kkma()

a = pd.DataFrame(columns=['text', 'score'])

for i, doc in enumerate(tfidfScorer(test['줄거리'])):
        print('document{}'.format(i))
        doc = pd.DataFrame(doc, columns=['word', 'score'])
        doc['title'] = test.loc[i, '영화이름']
        doc = doc[['title', 'word', 'score']]
        doc.drop_duplicates(subset=None, keep='first', inplace=True, ignore_index=False)
        doc = doc.sort_values(by='score', ascending=False)[:30]
        doc = doc.reset_index(); doc = doc.drop(labels='index', axis=1)
        
        cnt = 0
        for j in doc['word']:
            try:
                NN = kkma.nouns(j)[0]
                doc.loc[cnt, 'word'] = NN
            except IndexError:
                doc.drop(index=cnt, axis=0, inplace=True)
            cnt += 1

        if not os.path.exists('./movies.csv'):
            doc.to_csv('./movies.csv', encoding='utf-8-sig', mode='w', index=False)
        else:
            doc.to_csv('./movies.csv', encoding='utf-8-sig', mode='a', header=False, index=False)

