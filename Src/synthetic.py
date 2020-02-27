import random
import math
import numpy as np

def generate_baskets(items, num_baskets):
    basket_sizes = [math.floor(x) for x in (np.random.normal(6, 2, num_baskets))]
    new_baskets = [random.choices(items, k=n) for n in basket_sizes]
    return new_baskets