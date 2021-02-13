import pandas as pd
import itertools as it
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
from imblearn.over_sampling import SMOTE
from collections import Counter


def SMOTE_BEST_CV(fname, model, smote_bool=True, analysis=True, n_drop=0):
    '''
    Method to take in a dataset from a .csv file and run logistic regression

    pass a given model to be fitted and run within the code

    smote_bool is used to set SMOTE to fix unbalanced dataset issue
        this is defaulted to True for goal dataset

    n_drop refers to the number of features to drop from the dataset
        for goal dataset, this is manually configured to always drop is_promoted
    '''
    
    # Import the CSV file into Python
    A_data = pd.read_csv(fname)
    A_data = A_data.dropna()

    # features combinations selected
    removed_attributes = ['department', 'region', 'education', 'gender', 'recruitment_channel', 'no_of_trainings', 'age', 'previous_year_rating', 'length_of_service', 'awards_won?', 'avg_training_score']
    features_to_drop = list(it.combinations(removed_attributes, n_drop))

    best = (0, '')
    print('Number of iterations: {} '.format(len(features_to_drop)))
    for i, f in enumerate(features_to_drop):
        print('Iteration: {}'.format(i+1))

        # model fit
        f = list(f)
        f.append('is_promoted')
        X = A_data.drop(list(f), axis=1)
        y = (A_data.is_promoted == 1)
        
        if smote_bool:
            # transform the dataset
            oversample = SMOTE()
            X, y = oversample.fit_resample(X, y)

        # run a model
        model.fit(X, y)
        y_pred = cross_val_predict(model, X, y, cv=5)

        # tracking best model in terms of f1 score
        score = f1_score(y, y_pred)
        if score > best[0]:
            best = (score, f)
        print('Best F1_score: {} and removed attributes to acheive this: {}'.format(best[0], best[1]))

    # model fit with best
    X = A_data.drop(list(best[1]), axis=1)
    y = (A_data.is_promoted == 1)

    if smote_bool:
        # summarize the original distribution
        counter = Counter(y)
        print('Statistics before SMOTE: {}'.format(counter))

        # transform the dataset
        oversample = SMOTE()
        X, y = oversample.fit_resample(X, y)

        # summarize the new class distribution
        counter = Counter(y)
        print('Statistics after SMOTE: {}'.format(counter))

    # run best model
    model.fit(X, y)
    y_pred = cross_val_predict(model, X, y, cv=5)

    if analysis:
        # analysis
        print('Confusion Matrix: {}'.format(confusion_matrix(y, y_pred)))
        print('Recall: {}'.format(recall_score(y, y_pred)))
        print('Precision: {}'.format(precision_score(y, y_pred)))
        print('F1 Score: {}'.format(f1_score(y, y_pred)))
        