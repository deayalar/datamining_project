import pandas as pd
import textdistance

GROCERIES_FILE = '../Data/groceries/groceries.csv'
RECEIPES_FILE = '../Data/recipe-ingredients-dataset/train.json'

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