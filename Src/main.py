%reload_ext autoreload
%autoreload 2
%matplotlib inline

from recipes_analizer import RecipesAnalizer
from basket_processor import BasketProcessor
from plots import Plots

import numpy as np
from sklearn.metrics.pairwise import euclidean_distances

RECEIPES_FILE_TRAIN = '../Data/recipe-ingredients-dataset/train.json'
GROCERIES_FILE = '../Data/groceries/groceries.csv'
plot = Plots()

ra = RecipesAnalizer(source=RECEIPES_FILE_TRAIN)
recipes_df = ra.load_data()
unique_ingredients = ra.unique_ingredients(recipes_df['ingredients'])
recipes_df['ingredient_indexes'] = ra.replace_by_indexes(unique_ingredients, recipes_df['ingredients'])
tfidf_matrix = ra.tfidf_matrix(corpus=recipes_df['ingredient_indexes'])
recipes_df['cluster'], centroids = ra.cluster_recipes(tfidf_matrix, n_clusters=3)
plot.compare_cuisine(recipes_df)
vocabulary_text = [unique_ingredients[int(i)] for i in list(ra.vectorizer.vocabulary_.keys())]

bp = BasketProcessor()
baskets = bp.load_groceries_data(file=GROCERIES_FILE)
unique_items = list({item for basket in baskets for item in basket})
print('There are %d unique items' % len(unique_items)) #Includes all items in baskets
bp.find_common_items(vocabulary_text, unique_items) #Performs identical string comparison 
filtered_baskets = bp.get_filtered_baskets(baskets)
#Apply the fitted TfidfVectorizer to the baskets with 'transform' method
basket_corpus = [' '.join([str(unique_ingredients.index(item)) for item in basket]) for basket in filtered_baskets]
basket_tfidf = ra.vectorizer.transform(basket_corpus).todense() #No need to learn vocabulary
#Compute euclidean distances against centroids
distances = euclidean_distances(basket_tfidf, centroids) # baskets x n_clusters
basket_labels = [np.argmin(b) for b in distances]
plot.plot3d(distances=distances, basket_labels=basket_labels)