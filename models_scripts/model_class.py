"""
ECE143-Group9 Project WI20
Group Members: Samuel Cowin, Yejun Li, Armando Cadena, Jiazheng Bian, Kyle Janosky
"""

from collections import Counter
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, plot_confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC, SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.kernel_approximation import RBFSampler
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as smote_Pipeline
import matplotlib.pyplot as plt
import pandas as pd


class ClassificationModels:
    """
    Class to allow for customization of classification models on .csv datasets
    with printable analytics

    Example:
    Best results achieved on HR dataset (https://www.kaggle.com/arashnic/hr-ana) 
    after cleaning with the following lines of code:
        from model_class import ClassificationModels
        to_drop = ['department', 'gender']
        clf = ClassificationModels(C=1, gamma=0.5)
        clf.classification("[path to training data]/c_train.csv", clf.rbf_svm, to_drop, "is_promoted", True, True)

    Results:
        Statistics before SMOTE: Counter({False: 44428, True: 4232})
        Statistics after SMOTE: Counter({False: 44428, True: 44428})
        Confusion Matrix: [[6710 2268]
                           [1756 7038]]
        Recall: 0.8003183989083466
        Precision: 0.7562862669245648
        F1 Score: 0.7776795580110497
        Coefficients: ['region', 'education', 'recruitment_channel', 'no_of_trainings', \
                       'age', 'previous_year_rating', 'length_of_service', 'awards_won?', 'avg_training_score']
    """


    def __init__(self, C=1, penalty1='l1', penalty2='l2', loss='hinge', solver='liblinear', gamma=0.5):
        """
        Inputs
            C: float
                Taken from sklearn methods - 
                Inverse of regularization strength; must be a positive float.  
                Like in support vector machines, smaller values specify stronger  
                regularization.

            penalty1: str
                l1 penalty to pass to logistic regression

            penalty2: str
                l2 penalty to pass to SVM

            loss: str
                hinge loss to pass to SVM

            solver: str
                libinear loss to pass to linearSVC for fast computation

            gamma: float
                Taken from sklearn methods - 
                Kernel coefficient for 'rbf', 'poly' and 'sigmoid'. Similar to C in 
                that it is a regularization term
        """

        self.to_drop = ['department', 'gender']

        self.tuned_parameters = [
            {'kernel': ['poly'], 'C': [1, 10, 100, 1000], 'coef0': [1]}]

        self.log = LogisticRegression(
            penalty=penalty1,
            solver=solver,
            C=C
        )

        self.svm = Pipeline([
            ("scaler", StandardScaler()),
            ("SVM", LinearSVC(penalty=penalty2, C=C, loss=loss)),
        ])

        self.n_svm = Pipeline([
            ("scaler", StandardScaler()),
            ("SVM_nonlinear", SVC(kernel='poly', C=C, degree=3, coef0=1)),
        ])

        self.rbf_svm = Pipeline([
            ("scaler", StandardScaler()),
            ("SVM_nonlinear", SVC(kernel='rbf', C=C, gamma=gamma)),
        ])

        self.rbf_linear_svm = Pipeline([
            ("feature_map", RBFSampler(gamma=gamma, random_state=1, n_components=100)),
            ("scaler", StandardScaler()),
            ("SVM", LinearSVC(penalty=penalty2, C=C, loss=loss)),
        ])

        self.search_svm = GridSearchCV(
            SVC(), self.tuned_parameters, scoring='recall', verbose=10, n_jobs=-1, cv=3
        )

        self.NB = GaussianNB()

        self.MLP = MLPClassifier(random_state=1)


    def SMOTE_data(self, X, y, over='auto', under='auto', analysis=True):
        """
        Method to fix unbalanced datasets using SMOTE (https://arxiv.org/abs/1106.1813)

        The imbalanced-learn library is located here (https://github.com/scikit-learn-contrib/imbalanced-learn)

        Inputs
            X: Pandas dataframe
                Dataset features to be expanded

            y: Pandas series
                Target column from given dataset to be expanded

            over: str or float
                'auto' will implement default oversampling strategy while floating point numbers determine the 
                extent to how much to oversample. More information found on Github above

            under: str or float
                'auto' will implement default undersampling strategy while floating point numbers determine the 
                extent to how much to undersample. More information found on Github above

            analysis: bool
                Same parameter for classification method. Determines if analytics before and after SMOTE are to be 
                displayed
        """

        # summarize the original distribution
        counter = Counter(y)
        if analysis:
            print('Statistics before SMOTE: {}'.format(counter))

        # transform the dataset
        over = SMOTE(sampling_strategy=over)
        under = RandomUnderSampler(sampling_strategy=under)
        steps = [('o', over), ('u', under)]
        pipeline = smote_Pipeline(steps=steps)
        X, y = pipeline.fit_resample(X, y)

        # summarize the new class distribution
        counter = Counter(y)
        if analysis:
            print('Statistics after SMOTE: {}'.format(counter))

        return X, y


    def classification(self, fname, model, attributes_to_drop, target, smote_bool=True, analysis=True):
        """
        Environment to run fixed models with fixed attributes dropped

        Inputs
            fname: str
                .csv filename for extracting data
                File will undergo feature removal and NaN removal before model fitting

            model: sklearn classification model
                model for fitting the data. Needs to have fit() and predict() methods

            attributes_to_drop: list[str]
                attributes to drop from the dataset if so desired

            target: str
                target attribute within the dataset that will also be dropped from the training data

            smote_bool: bool
                Indication if SMOTE will be applied to the dataset. This is useful to offset the defecits 
                of an unbalanced dataset as was done with the example in the class description

            analysis: bool
                Indication if analytics and statistics should be computed and displayed for the model fit
        """

        # Import the CSV file into Python
        A_data = pd.read_csv(fname)
        A_data = A_data.dropna()

        # Set the training data and target data
        attributes_to_drop.append(target)
        X = A_data.drop(attributes_to_drop, axis=1)
        y = (A_data[target] == 1)

        if smote_bool:
            X, y = self.SMOTE_data(
                X, y, over='auto', under='auto', analysis=analysis)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42)

        try:
            model.fit(X_train, y_train)
            y_pred = model.cross_val_predict(X_test, cv=5)
        except AttributeError: 
            y_pred = model.fit(X_train, y_train).predict(X_test)

        if analysis:
            # {None, 'true', 'pred', 'all'}
            plot_confusion_matrix(model, X_test, y_test, normalize=None)
            plt.show()
            print('Confusion Matrix: {}'.format(
                confusion_matrix(y_test, y_pred)))
            print('Recall: {}'.format(recall_score(y_test, y_pred)))
            print('Precision: {}'.format(precision_score(y_test, y_pred)))
            print('F1 Score: {}'.format(f1_score(y_test, y_pred)))
            print('Coefficients: {}'.format(list(X.columns)))
            try:
                print('Values: {}'.format(model.coef_))
            except AttributeError:
                pass
