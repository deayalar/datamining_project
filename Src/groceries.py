import pandas as pd
import textdistance

GROCERIES_FILE = 'Data/groceries/groceries.csv'
RECEIPES_FILE = 'Data/recipe-ingredients-dataset/train.json'

#Groceries dataset reading
baskets = []
with open(GROCERIES_FILE, 'r', encoding='utf-8') as groceries_file:
    lines = groceries_file.read().splitlines()
    baskets = [l.split(',') for l in lines]
len(baskets)

unique_items = {item for basket in baskets for item in basket}
unique_items = list(unique_items)
len(unique_items)

receipes_df = pd.read_json(RECEIPES_FILE)
receipes_df.head()

unique_ingredients = set()
for row in receipes_df['ingredients']:
        unique_ingredients.update(row)
len(unique_ingredients)

discarded_indexes = [item not in unique_ingredients for item in unique_items]

discarded_items = [unique_items[i] for i in range(0,len(discarded_indexes)) if discarded_indexes[i] == True]
kept_items = [unique_items[i] for i in range(0,len(discarded_indexes)) if discarded_indexes[i] == False]

len(discarded_items) + len(kept_items) == len(discarded_indexes)

#TODO: Find a better way to associate existing items in the basket 
text1 = "semi-finished bread"
text2 = "spread cheese"
#Optional similarity for strings
print('levenshtein --> ' + str(textdistance.levenshtein.normalized_similarity(text1, text2)))
print('hamming --> ' + str(textdistance.hamming.normalized_similarity(text1, text2)))
print('jaro_winkler --> ' + str(textdistance.jaro_winkler(text1, text2)))
print('jaccard --> ' + str(textdistance.jaccard(text1 , text2)))

#Filter the baskets with the ingredients-related dataset
filtered_baskets = []
for basket in baskets:
    filtered_basket = []
    for item in basket:
        if item in kept_items:
            filtered_basket.append(item)
    filtered_baskets.append(filtered_basket)


#Association rules ENTIRE dataset
from mlxtend.frequent_patterns import apriori
from mlxtend.preprocessing import TransactionEncoder

te = TransactionEncoder()
#dataset = receipes_df['ingredients'].values #For all cuisine types
te_ary = te.fit(baskets).transform(baskets)
df = pd.DataFrame(te_ary, columns=te.columns_)
df.columns.values
df

apriori_df = apriori(df, min_support=0.05, use_colnames=True)
apriori_df.head(100)

#Association rules FILTERED dataset (IT gives exactly the same values, but)
te = TransactionEncoder()
#dataset = receipes_df['ingredients'].values #For all cuisine types
te_ary = te.fit(filtered_baskets).transform(filtered_baskets)
df = pd.DataFrame(te_ary, columns=te.columns_)
df.columns.values
df

apriori_df = apriori(df, min_support=0.05, use_colnames=True)
apriori_df.head(100)

#How many times the definitive items appear as ingredients in the receipes
ingredients = list()
for row in receipes_df['ingredients']:
        ingredients.append(row)
len(ingredients)

ingredients_count = ingredients_count(ingredients)

for i in kept_items:
    print(i + ' ' + str(ingredients_count.get(i)))

#Build a matrix which to store the jaccard simmilarity between the receipes and the baskets, whit no filters in receipes
#Use the filtered baskets
 
import numpy as np
#simmilarities = np.zeros((len(ingredients), len(kept_items)))
#simmilarities.shape

simmilarities = list()

print('jaccard --> ' + str())
for i in ingredients:
    print(i)
    for j in filtered_baskets:
        simmilarities.append(textdistance.jaccard(i , j))

sim_array = np.asarray(simmilarities).reshape(len(ingredients), len(kept_items))
np.savetxt('sim.csv', sim_array, delimiter=',')
#Nothing useful the above matrix takes a long time to be computed Definitely clean the receipes dataset
#Try a cluster for groceries

######
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
import re
receipes_df['ingredients_clean_string'] = [' , '.join(z).strip() for z in receipes_df['ingredients']]  
receipes_df['ingredients_string'] = [' '.join([WordNetLemmatizer().lemmatize(re.sub('[^A-Za-z]', ' ', line)) for line in lists]).strip() for lists in receipes_df['ingredients']]    

corpustr = receipes_df['ingredients_string']
vectorizertr = TfidfVectorizer(stop_words='english',
                             ngram_range = ( 1 , 1 ),analyzer="word", 
                             max_df = .57 , binary=False , token_pattern=r'\w+' , sublinear_tf=False)

tfidftr=vectorizertr.fit_transform(corpustr).todense()
tfidftr.shape

from sklearn.decomposition import TruncatedSVD
svd = TruncatedSVD(n_components=100)
tfidftr_reduced = svd.fit_transform(tfidftr)

from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=4, random_state=0).fit(tfidftr)
kmeans.labels_
receipes_df['cluster'] = kmeans.labels_

#uniqueValues, occurCount = np.unique(kmeans.labels_, return_counts=True)
#print("Unique Values : " , uniqueValues)
#print("Occurrence Count : ", occurCount)

#For eah group, plot an histogram of cuisine types, just to validate that the cluster makes sense
df = receipes_df.groupby(['cuisine', 'cluster']).size().unstack(fill_value=0)
df.sum()


################################ https://stevenloria.com/tf-idf/

import math
from textblob import TextBlob as tb

document1 = tb("""Python is a 2000 made-for-TV horror movie directed by Richard
Clabaugh. The film features several cult favorite actors, including William
Zabka of The Karate Kid fame, Wil Wheaton, Casper Van Dien, Jenny McCarthy,
Keith Coogan, Robert Englund (best known for his role as Freddy Krueger in the
A Nightmare on Elm Street series of films), Dana Barron, David Bowe, and Sean
Whalen. The film concerns a genetically engineered snake, a python, that
escapes and unleashes itself on a small town. It includes the classic final
girl scenario evident in films like Friday the 13th. It was filmed in Los Angeles,
 California and Malibu, California. Python was followed by two sequels: Python
 II (2002) and Boa vs. Python (2004), both also made-for-TV films.""")

document2 = tb("""Python, from the Greek word (πύθων/πύθωνας), is a genus of
nonvenomous pythons[2] found in Africa and Asia. Currently, 7 species are
recognised.[2] A member of this genus, P. reticulatus, is among the longest
snakes known.""")

document3 = tb("""The Colt Python is a .357 Magnum caliber revolver formerly
manufactured by Colt's Manufacturing Company of Hartford, Connecticut.
It is sometimes referred to as a "Combat Magnum".[1] It was first introduced
in 1955, the same year as Smith &amp; Wesson's M29 .44 Magnum. The now discontinued
Colt Python targeted the premium revolver market segment. Some firearm
collectors and writers such as Jeff Cooper, Ian V. Hogg, Chuck Hawks, Leroy
Thompson, Renee Smeets and Martin Dougherty have described the Python as the
finest production revolver ever made.""")

bloblist = [document1, document2, document3]
for i, blob in enumerate(bloblist):
    print("Top words in document {}".format(i + 1))
    scores = {word: tfidf(word, blob, bloblist) for word in blob.words}
    sorted_words = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    for word, score in sorted_words[:3]:
        print("\tWord: {}, TF-IDF: {}".format(word, round(score, 5)))


import math
from textblob import TextBlob as tb

def tf(word, blob):
    return blob.words.count(word) / len(blob.words)

def n_containing(word, bloblist):
    return sum(1 for blob in bloblist if word in blob.words)

def idf(word, bloblist):
    return math.log(len(bloblist) / (1 + n_containing(word, bloblist)))

def tfidf(word, blob, bloblist):
    return tf(word, blob) * idf(word, bloblist)