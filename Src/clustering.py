%matplotlib inline
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import CountVectorizer

import summarizer as sm

RECEIPES_FILE = '../Data/recipe-ingredients-dataset/train.json'

receipes_df = pd.read_json(RECEIPES_FILE)
receipes_df.head()
receipes_df['cuisine'].nunique()

#Set of unique ingredients
#TODO: List comprehension can be used here
ingredients = set()
for row in receipes_df['ingredients']:
        ingredients.update(row)
len(ingredients)
len(receipes_df)

#Get a sorted dictionary of unique ingredients and it's frecuency in receipes
ingredients_count = sm.ingredients_count(receipes_df['ingredients'])
len(ingredients_count)

#Plot number of ingredients vs frecuency threshold
#TODO: Move this a module
def plot_ingredients_frecuency(ingredients_count, limit=15):
    filtered_count = {}
    for i in range(0, limit):
        filtered_count[i] = len(sm.filter_ingredients(ingredients_count, min_frecuency=i))
    plt.scatter(*zip(*filtered_count.items()))
    plt.show()

plot_ingredients_frecuency(ingredients_count)
threshold_frecuency = 15 #Decrease of the number of ingredients in almost a half

#Filter by threshold_frecuency thos ingredients that appear in more than _threshold_frecuency] receipes
ingredients_filtered = sm.filter_ingredients(ingredients_count, min_frecuency=threshold_frecuency)
len(ingredients_filtered)

#Create text of ingredients to compute the term frecuency matrix
ingredients_text = []
for i in receipes_df['ingredients']:
    i = ' '.join(i)
    ingredients_text.append(i)
receipes_df['ingredients_text'] = ingredients_text

#Frecuency term matrix
ingredients_list = list()
for i in receipes_df['ingredients_text']:
    ingredients_list.append(i)

vectorizer = CountVectorizer(vocabulary=list(ingredients_filtered.keys()))
frecuency_matrix = vectorizer.fit_transform(ingredients_list)
print(len(vectorizer.vocabulary_))
# summarize frecuency matrix
print(frecuency_matrix.shape)
print(frecuency_matrix.toarray())

#Join frecuency matrix to dataframe
receipes_df = receipes_df.join(pd.DataFrame(frecuency_matrix.toarray()))
receipes_df.head()

v = vectorizer.transform([receipes_df.iloc[39773,3]])
print(v.toarray()[0])

def get_ingredient(index, vocabulary): 
    for k, val in vocabulary.items(): 
        if val == index:
            print(k)

get_ingredient(0, vectorizer.vocabulary_)
########################## TF IDF + Clustering ############################

from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(vocabulary=list(ingredients_filtered.keys()))
frecuency_matrix = vectorizer.fit_transform(ingredients_list)
print(len(vectorizer.vocabulary_))
# summarize frecuency matrix
print(frecuency_matrix.shape)
print(frecuency_matrix.toarray())
receipes_df = receipes_df.join(pd.DataFrame(frecuency_matrix.toarray()))

from sklearn.cluster import AgglomerativeClustering
clustering = AgglomerativeClustering().fit(frecuency_matrix.toarray())

from sklearn.decomposition import PCA
pca = PCA(n_components=2)
# print(pca.explained_variance_ratio_)   
reduced_data = pca.fit_transform(frecuency_matrix)

receipes_df.drop([], axis=1)

X = np.array([[1, 2], [1, 4], [1, 0], [4, 2], [4, 4], [4, 0]])
X.shape
clustering = AgglomerativeClustering(distance_threshold = 0.5).fit(X)
clustering
clustering.distance_threshold

from sklearn.cluster import KMeans
import numpy as np

kmeans = KMeans(n_clusters=10, random_state=0).fit(frecuency_matrix)