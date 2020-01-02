#Returns a dictionary of unique ingredients and frecuency in receipes sorted by frecuency
def ingredients_count(receipe_ingredients):
    #TODO Replace by:
    #ingredients_count = {k: v for k, v in sorted(dict(Counter(ingredients)).items(), key=lambda item: item[1], reverse=True)}
    ingredients_count = dict() #Can be used a Counter https://docs.python.org/2/library/collections.html#collections.Counter
    for row in receipe_ingredients:
        for i in row:
            ingredients_count[i] = ingredients_count.get(i, 0) + 1
    ingredients_count = {k: v for k, v in sorted(ingredients_count.items(), key=lambda item: item[1], reverse = True)}
    return ingredients_count

#Filter the ingredients
#ingredients_count: Dictionary of ingredients count
#min_frecuency: Lower limit of frecuency
def filter_ingredients(ingredients_count, min_frecuency=15):
    ingredients_filtered = dict(filter(lambda x: x[1] > min_frecuency, ingredients_count.items()))
    return ingredients_filtered