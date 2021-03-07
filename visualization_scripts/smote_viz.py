from collections import Counter
from sklearn.datasets import make_classification
from imblearn.over_sampling import SMOTE
from matplotlib import colors
from matplotlib import pyplot
from numpy import where
import numpy as np

X, y = make_classification(n_samples=10000, n_features=2, n_redundant=0,
        n_clusters_per_class=1, weights=[0.99], flip_y=0, random_state=1)
# oversample = SMOTE()
# X, y = oversample.fit_resample(X, y)
counter = Counter(y)

for label, _ in counter.items():
    row_ix = where(y == label)[0]
    colors = np.array(['gold', 'lemonchiffon'])
    z = np.zeros(9900, dtype='int')
    o = np.ones(100, dtype='int')
    color = [z, o]
    index = color[label]
    pyplot.scatter(X[row_ix, 0], X[row_ix, 1], c=colors[index])
    ax = pyplot.gca()
    ax.set_facecolor((0.26, 0.26, 0.26))
pyplot.show()