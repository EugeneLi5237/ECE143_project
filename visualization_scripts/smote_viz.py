from collections import Counter
from sklearn.datasets import make_classification
from imblearn.over_sampling import SMOTE
from matplotlib import pyplot
from numpy import where
import numpy as np


def smote_visualization(n=10000, nz=9900, no=100, smote_bool=False):
    """
    Method to visualize dataset modifications made by SMOTE algorithm
    """

    assert isinstance(n, int)
    assert isinstance(nz, int)
    assert isinstance(no, int)
    assert n > nz >= no > 0
    assert isinstance(smote_bool, bool)

    X, y = make_classification(n_samples=n, n_features=2, n_redundant=0,
                               n_clusters_per_class=1, weights=[0.99], flip_y=0, random_state=1)
    if smote_bool:
        oversample = SMOTE()
        X, y = oversample.fit_resample(X, y)
    counter = Counter(y)

    for label, _ in counter.items():
        row_ix = where(y == label)[0]
        colors = np.array(['gold', 'lemonchiffon'])
        z = np.zeros(nz, dtype='int')
        o = np.ones(no, dtype='int')
        color = [z, o]
        index = color[label]
        pyplot.scatter(X[row_ix, 0], X[row_ix, 1], c=colors[index])
        ax = pyplot.gca()
        ax.set_facecolor((0.26, 0.26, 0.26))
    pyplot.show()
