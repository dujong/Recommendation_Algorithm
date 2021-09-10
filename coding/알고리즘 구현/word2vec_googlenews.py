from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import manhattan_distances
import pandas as pd
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
import gensim

train = pd.read_csv('알고리즘 구현/movies.csv', engine='python')
train_ = train['word']
items = train_[:397]

model = gensim.models.KeyedVectors.load_word2vec_format('googlenews/GoogleNews-vectors-negative300.bin', binary=True)

item_vectors = [(item, model[item]) for item in items if item in model]
print('item_vector 개수 :{}'.format(len(item_vectors)))

vectors = np.asanyarray([x[1] for x in item_vectors])

tsne = TSNE(n_components=2, perplexity=10, verbose=2).fit_transform(vectors)

x = tsne[:,0]
y = tsne[:,1]
fig,ax = plt.subplots()
ax.scatter(x,y)
for item, x1, y1 in zip(item_vectors, x, y):
    ax.annotate(item[0],(x1,y1),size=14)

plt.show()

# sent1 = test['줄거리'][0]
# sent2 = test['줄거리'][50]

# tfidf_vectorizer = TfidfVectorizer()
# tfidf_matrix_1 = tfidf_vectorizer.fit_transform([sent1, sent2])
# idf = tfidf_vectorizer.idf_

# a = manhattan_distances(tfidf_matrix_1[0], tfidf_matrix_1[1])
# print(a)

