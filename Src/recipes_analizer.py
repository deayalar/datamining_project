import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

class RecipesAnalizer:

    def __init__(self, source):
        self.source = source
    
    def load_data(self):
        recipes_df = pd.read_json(self.source).drop(['id'], axis=1)
        return recipes_df

    def unique_ingredients(self, recipes):
        unique_ingredients = set()
        for row in recipes:
                unique_ingredients.update(row)
        unique_ingredients = list(unique_ingredients)
        print("Created a list of %d unique ingredients from %d recipes" % (len(unique_ingredients), len(recipes)))
        return unique_ingredients

    def replace_by_indexes(self, unique_ingredients, recipes):
        """Convert the ingredients to indexes in the unique_ingredients list
        Parameters
        ----------
        unique_ingredients : list
            List of unique ingredients
        recipes : list
            List of recipes with ingredients as text
        Returns
        -------
        List
            List of recipes with indexes instead of ingredients
        """
        ing_index_array = [[unique_ingredients.index(line) for line in lists] for lists in recipes]
        ing_index_plain = [str(z).replace(',', '').strip('\[').strip('\]') for z in ing_index_array]
        return ing_index_plain

    def tfidf_matrix(self, corpus):
        self.vectorizer = TfidfVectorizer(analyzer="word", min_df=.01,
                                    max_df = .6 , binary=False , token_pattern=r'\w+' , sublinear_tf=False)
        tfidf_matrix = self.vectorizer.fit_transform(corpus).todense()
        print('Tf-idf vectorizer has selected %d ingredients' % tfidf_matrix.shape[1])
        return tfidf_matrix

    #TODO; Extend this method to allow other cluster algorithms
    def cluster_recipes(self, data, n_clusters=4):
        kmeans = KMeans(n_clusters=n_clusters, random_state=42).fit(data)
        return (kmeans.labels_, kmeans.cluster_centers_)