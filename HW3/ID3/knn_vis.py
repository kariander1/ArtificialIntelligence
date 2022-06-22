# import matplotlib.pyplot as plt
# import numpy as np
# import pandas as pd
# from sklearn import datasets, neighbors
# from mlxtend.plotting import plot_decision_regions
#
#
# def knn_comparison(x, y, k):
#  clf = neighbors.KNeighborsClassifier(n_neighbors=k)
#  clf.fit(x, y)
#  # Plotting decision region
#  plot_decision_regions(x, y, clf=clf, legend=2,)
#  # Adding axes annotations
#  plt.xlabel('X')
#  plt.ylabel('Y')
#  plt.title('Knn with K=' + str(k))
#  plt.show()
#
#
# x = np.array([(1, 5), (2, 6), (3, 7), (4, 8), (5, 9), (7, 2), (8, 3), (2, 7), (3, 8), (5, 1), (6, 2), (7, 3), (8, 4), (9, 5)])
# y = np.array([0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1])
# knn_comparison(x, y, 1)

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap
from sklearn import neighbors, datasets
from sklearn.inspection import DecisionBoundaryDisplay
import numpy as np

n_neighbors = 4

# import some data to play with
iris = datasets.load_iris()

# we only take the first two features. We could avoid this ugly
# slicing by using a two-dim dataset

x = np.array([(1, 5), (2, 6), (3, 7), (4, 8), (5, 9), (7, 2), (8, 3), (2, 7), (3, 8), (5, 1), (6, 2), (7, 3), (8, 4), (9, 5)])
y = np.array([0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1])

# Create color maps
cmap_light = ListedColormap(["orange", "blue"])
cmap_bold = ["darkorange", "c"]

for weights in ["uniform", "distance"]:
    # we create an instance of Neighbours Classifier and fit the data.
    clf = neighbors.KNeighborsClassifier(n_neighbors, weights=weights)
    clf.fit(x, y)

    _, ax = plt.subplots()
    DecisionBoundaryDisplay.from_estimator(
        clf,
        x,
        cmap=cmap_light,
        ax=ax,
        response_method="predict",
        plot_method="pcolormesh",
        xlabel='X',
        ylabel='Y',
        shading="auto",
    )

    # Plot also the training points
    sns.scatterplot(
        x=x[:, 0],
        y=x[:, 1],
        hue=y,
        palette=cmap_bold,
        alpha=1.0,
        edgecolor="black",
    )
    plt.title(
        "2-Class classification (k = %i)" % (n_neighbors)
    )

plt.show()