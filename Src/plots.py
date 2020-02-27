import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import pandas as pd
import numpy as np

class Plots:

    def __init__(self):
        self.colors = ['red', 'blue', 'green', 'yellow']
        self.hr_labels = ['asian', 'english', 'western']

    def plot3d(self, distances, basket_labels):
        #3D plot of clusters
        #svd = TruncatedSVD(n_components=3)
        #distances_reduced_3d = svd.fit_transform(distances)
        distances_reduced_3d = distances
        distances_df = pd.DataFrame(distances_reduced_3d, columns=('x', 'y', 'z'))
        distances_df['cluster'] = basket_labels

        colors=['red', 'blue', 'green', 'yellow']
        hr_labels = ['asian', 'english', 'western']
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        for i, cluster in distances_df.groupby('cluster'):
            _ = ax.scatter(cluster['x'], cluster['y'], cluster['z'], c=colors[i], label=hr_labels[i])
        ax.legend()
        plt.show()

    def compare_cuisine(self, recipes_df):
        cuisine_df = recipes_df.groupby(['cuisine', 'cluster']).size().unstack(fill_value=0)
        cuisine_df['label'] = cuisine_df.idxmax(axis=1)
        print(cuisine_df[cuisine_df['label'] == 0]['label'])
        print(cuisine_df[cuisine_df['label'] == 1]['label'])
        print(cuisine_df[cuisine_df['label'] == 2]['label'])

        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        for i in range(0,3):
            ax.scatter(cuisine_df[cuisine_df['label'] == i][0], cuisine_df[cuisine_df['label'] == i][1],cuisine_df[cuisine_df['label'] == i][2], c=self.colors[i], label=self.hr_labels[i])
        ax.legend()
        plt.show()