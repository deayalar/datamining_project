# datamining_project
Data Mining Project 19/20 Unitn

This repository contains the final artifacts of the
project of Data Mining for the academic year
2019/2020 at the University of Trento. 

This project is based on two unrelated
datasets from Kaggle competences. The first dataset
is a JSON file that contains a list of recipes with their
corresponding ingredients and cuisine types https://www.kaggle.com/kaggle/recipe-ingredients-dataset. 
On the other hand, the second dataset contains grocery
market basket data in CSV format https://www.kaggle.com/irfanasrullah/groceries, this is essentially
a set of baskets with their corresponding list of
items. 

The goal of this project is to group the
market customers based on not only on what they
are buying but what they are cooking based on the
recipes dataset. To achieve this goal, recipes and
baskets are treated as documents with a set of
words. So, clusters of recipes have been created
using k-means with previous processing using TF-
IDF(Term Frequency-Inverse Document Frequency).
On the other side, after removing all the items that
do not correspond to any recipe, TF-IDF
preprocessing was applied to the market basket
dataset using the same vocabulary used for recipes.
Finally, each basket was associated with the closest
centroid of recipe clusters. In that way, the result is
a classification of customers based on what they are
buying related to the recipes dataset. At the end
each cluster is labeled with a human-understandable
text to interpret the results
