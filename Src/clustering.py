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

################################# CLUSTERING OF RECEIPES ########################################
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import pandas as pd
import numpy as np
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.decomposition import TruncatedSVD

RECEIPES_FILE = '../Data/recipe-ingredients-dataset/train.json'

receipes_df = pd.read_json(RECEIPES_FILE)
receipes_df.head()

unique_ingredients = set()
for row in receipes_df['ingredients']:
        unique_ingredients.update(row)
unique_ingredients = list(unique_ingredients)
len(unique_ingredients)

#receipes_df['ingredients_string'] = [' '.join(z).strip() for z in receipes_df['ingredients']]
#receipes_df['ingredients_string_x'] = [[re.sub('[^A-Za-z- ]', '', line).strip().lower() for line in lists] for lists in receipes_df['ingredients']]

receipes_df['ing_index_array'] = [[unique_ingredients.index(line) for line in lists] for lists in receipes_df['ingredients']]
receipes_df['ing_index_plain'] = [str(z).replace(',', '').strip('\[').strip('\]') for z in receipes_df['ing_index_array']]
receipes_df.drop("ingredients", inplace=True, axis=1)

vectorizer = TfidfVectorizer(analyzer="word", min_df=.01,
                             max_df = .6 , binary=False , token_pattern=r'\w+' , sublinear_tf=False)

corpus = receipes_df['ing_index_plain']
tfidf_matrix = vectorizer.fit_transform(corpus).todense()
tfidf_matrix.shape

clusters = 4
kmeans = KMeans(n_clusters=clusters, random_state=0).fit(tfidf_matrix)
receipes_df['cluster'] = kmeans.labels_

#This is just to validate that the cluster makes sense with cuisine types, BUT cuisine is not considered in the algorithm
cuisine_df = receipes_df.groupby(['cuisine', 'cluster']).size().unstack(fill_value=0)
cuisine_df
cuisine_df.sum()

vocabulary_text = [unique_ingredients[int(i)] for i in list(vectorizer.vocabulary_.keys())]
centroids = kmeans.cluster_centers_

#--------------------------------------------- GROCERIES BASKETS -------------------------------------------------
GROCERIES_FILE = '../Data/groceries/groceries.csv'
#Groceries dataset reading
baskets = []
with open(GROCERIES_FILE, 'r', encoding='utf-8') as groceries_file:
    lines = groceries_file.read().splitlines()
    baskets = [l.split(',') for l in lines]
len(baskets)

unique_items = list({item for basket in baskets for item in basket})
len(unique_items) #Includes all ingredients

#TODO: This is comparing exact values (Use similarities to include similar e.g. bottled water = water, whipped/sour cream = whiped sour cream) Use any simmilarity to improve this
discarded_indexes = [item not in vocabulary_text for item in unique_items]
discarded_items = [unique_items[i] for i in range(0,len(discarded_indexes)) if discarded_indexes[i] == True]
kept_items = [unique_items[i] for i in range(0,len(discarded_indexes)) if discarded_indexes[i] == False]
len(discarded_items) + len(kept_items) == len(discarded_indexes)

#filter baskets with the receipes vocabulary
filtered_baskets = []
for basket in baskets:
    filtered_basket = []
    for item in basket:
        if item in kept_items:
            filtered_basket.append(item)
    if filtered_basket:
        filtered_baskets.append(filtered_basket)
len(filtered_baskets)
#apply the fitted TfidfVectorizer to the baskets
#transform 
basket_index = [' '.join([str(unique_ingredients.index(item)) for item in basket]) for basket in filtered_baskets]
basket_tfidf = vectorizer.transform(basket_index).todense() #No need to learn vocabulary

#Compute euclidean distances against centroids
distances = euclidean_distances(basket_tfidf, centroids)
distances.shape
basket_labels = [np.argmin(b) for b in distances]
np.unique(basket_labels)

#---------------------------------------------- PLOTING BASKET GROUPS -----------------------------------------
#Plot clustered baskets
svd = TruncatedSVD(n_components=2)
distances_reduced = svd.fit_transform(distances)
distances_reduced.shape
x, y = zip(*distances_reduced)
plt.scatter(x, y, c=basket_labels)
plt.show()

#3D plot of clusters
svd = TruncatedSVD(n_components=3)
distances_reduced_3d = svd.fit_transform(distances)

distances_df = pd.DataFrame(distances_reduced_3d, columns=('x', 'y', 'z'))
distances_df['cluster'] = basket_labels

colors=['red', 'blue', 'green', 'yellow']
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
for i, cluster in distances_df.groupby('cluster'):
    _ = ax.scatter(cluster['x'], cluster['y'], cluster['z'], c=colors[i], label=i)
ax.legend()
plt.show()