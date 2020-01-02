#This script is intended to get frequent itemsets and rule associations
# -> Remove specific products
# -> Remove (    oz.) 
# -> 2% reduced-fat milk
# black-eyed peas
import json
import pandas as pd
from collections import Counter

RECEIPES_FILE = '../Data/recipe-ingredients-dataset/train.json'

receipes_df = pd.read_json(RECEIPES_FILE)
receipes_df.head()

receipes_df['cuisine'].unique()
receipes_df.groupby(['cuisine']).count()

#Set of unique ingredients
ingredients = set()
for row in receipes_df['ingredients']:
        ingredients.update(row)
len(ingredients)

#Summarize information of ingredients per cuisine type
receipes_df['number_ingredients'] = receipes_df['ingredients'].apply(len)
receipes_by_cuisine = (receipes_df[['ingredients', 'cuisine']].groupby(['cuisine']).count())
receipes_by_cuisine.rename(columns={'ingredients':'number_receipes'}, inplace=True)

ingredients_by_cuisine = receipes_df[['cuisine', 'ingredients']].groupby(['cuisine']).sum()
ingredients_by_cuisine['ingredients_with_rep'] = ingredients_by_cuisine['ingredients'].apply(len)
ingredients_by_cuisine['unique_ingredients'] = ingredients_by_cuisine['ingredients'].apply(lambda ing: dict(Counter(ing)))
ingredients_by_cuisine['count'] = ingredients_by_cuisine['unique_ingredients'].apply(lambda counter: len(counter))
ingredients_by_cuisine = ingredients_by_cuisine.join(receipes_by_cuisine)
#ingredients_by_cuisine.drop('count', axis = 1, inplace = True)

#ASSOCIATIION RULES PER CUISINE Example with brazilian cuisine
cu_df = receipes_df[receipes_df['cuisine'] == 'indian']

from mlxtend.frequent_patterns import apriori
from mlxtend.preprocessing import TransactionEncoder

te = TransactionEncoder()
dataset = cu_df['ingredients'].values #For a specific cuisine
#dataset = receipes_df['ingredients'].values #For all cuisine types

te_ary = te.fit(dataset).transform(dataset)
df = pd.DataFrame(te_ary, columns=te.columns_)
df.columns.values
df

apriori(df, min_support=0.1, use_colnames=True)

#Which ingredients characterize each type of cuisine
#Frecuence of ingredients each type of cuisine
cuisine = "greek"
ingredients_map = dict(ingredients_by_cuisine.loc[ingredients_by_cuisine.index == cuisine, 'unique_ingredients'].get(cuisine))
ingredients_map = {k: v for k, v in sorted(ingredients_map.items(), key=lambda item: item[1], reverse=True)}
number_receipes = ingredients_by_cuisine.loc[ingredients_by_cuisine.index == cuisine, 'number_receipes'].get(cuisine)
def ingredients_ratio(frecuency, receipes):
    return frecuency / receipes
ingredients_map.update({k: ingredients_ratio(v, number_receipes) for k, v in ingredients_map.items()})
list(ingredients_map.items())[0:100]


#TF_IDF of ingredients per cuisine, this penalize most common ingredients
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(vocabulary=list(ingredients_map.keys()))
cuisine_receipes = receipes_df[receipes_df['cuisine'] == cuisine

#Create text of ingredients to compute the term frecuency matrix
ingredients_text = []
for i in cuisine_receipes['ingredients']:
    i = ' '.join(i)
    ingredients_text.append(i)
cuisine_receipes['ingredients_text'] = ingredients_text
ingredients_by_cuisine.loc[cuisine]
receipes_corpus = cuisine_receipes['ingredients_text'].values

len(receipes_corpus)

frecuency_matrix = vectorizer.fit_transform(receipes_corpus)
print(len(vectorizer.vocabulary_))
# summarize frecuency matrix
print(frecuency_matrix.shape)
tfidf_matrix = frecuency_matrix.toarray()
print(frecuency_matrix.toarray())
#receipes_df = receipes_df.join(pd.DataFrame(frecuency_matrix.toarray()))

import numpy as np
receipes_corpus[2]
relevant_ingredients = np.argwhere(tfidf_matrix[2] != 0)

print(list(ingredients_map.keys())[8])
print(list(vectorizer.vocabulary_.keys())[list(vectorizer.vocabulary_.values()).index(8)])
import matplotlib.pyplot as plt
means = np.mean(tfidf_matrix, axis=1)
plt.hist(means)
plt.show()


np.argwhere(means <= 0.0005)
print(list(vectorizer.vocabulary_.keys())[list(vectorizer.vocabulary_.values()).index(1158)])
ingredients_by_cuisine.loc[cuisine]['unique_ingredients'].get('salt')

means[vectorizer.vocabulary_.get('salt')]

cuisine_receipes['ingredients_text'][0]