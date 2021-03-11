# HR Prediction Analysis for Promotions

ECE143 WI21 Group 9

## Requirements:
Use python 3.8+ <br /><br />
numpy==1.19.4 <br />
imbalanced_learn==0.7.0<br />
pandas==1.0.5<br />
seaborn==0.11.1<br />
matplotlib==3.2.2<br />
imblearn==0.0<br />
scikit_learn==0.24.1<br />

## File Structure
* Raw data and cleaned data are included 
* The Models Class that generated performance data for our presentation is included
* Plots used within the presentation are included
* Project proposal stating what is to be accomplished with the project
* Scripts to generate the plots are included 
* Requirements are included in a text file
* Full Kaggle Notebook of our visualization and implementations is included
* Group assignment developing validation and functional testing for homework assignment is included

## Running Model Implementations
Example:<br />
Import the models class for accessing linear and nonlinear models<br />
```python
from model_class import ClassificationModels
```
List the features you wish to drop from the feature space (not including target)<br />
```python
to_drop = ['department', 'gender']
```
Initialize the model class with certain hyperparameters<br />
```python
clf = ClassificationModels(C=1, gamma=0.5)
```
Provide the path to training data, decide which model to use, and provide the target<br />
```python
clf.classification("[path to training data]/c_train.csv", clf.rbf_svm, to_drop, "is_promoted", True, True)
```
