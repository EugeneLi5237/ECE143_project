from sklearn.linear_model import LogisticRegression
from models_utils import SMOTE_BEST_CV

log = LogisticRegression(
    penalty='l1',
    solver='liblinear',
    C=0.5,
    max_iter=10000)

SMOTE_BEST_CV("c_train.csv", log, smote_bool=True, analysis=True, n_drop=0)