# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 15:28:43 2020

@author: GENIUS
"""
# Naive Bayes

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('train.csv')
X_train = dataset.iloc[:, [2,4,5,6,7,9]].values
y_train = dataset.iloc[:, 1].values

from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(X_train[:, [2]])
X_train[:, [2]] = imputer.transform(X_train[:, [2]])

# Encoding categorical data
# Encoding the Independent Variable
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X_train[:, 6] = labelencoder_X.fit_transform(X_train[:, 6])

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X_train[:, 1] = labelencoder_X.fit_transform(X_train[:, 1])
onehotencoder = OneHotEncoder(categorical_features = [1])
X_train = onehotencoder.fit_transform(X_train).toarray()

onehotencoder = OneHotEncoder(categorical_features = [7])
X_train = onehotencoder.fit_transform(X_train).toarray()

# Encoding the Dependent Variable

# creating the test sets
dataset_1 = pd.read_csv('test.csv')
X_test = dataset_1.iloc[:, [1,3,4,5,6,8,10]].values


from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(X_test[:, [2]])
X_test[:, [2]] = imputer.transform(X_test[:, [2]])

# Encoding categorical data
# Encoding the Independent Variable
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X_test[:, 1] = labelencoder_X.fit_transform(X_test[:, 1])

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X_test[:, 6] = labelencoder_X.fit_transform(X_test[:, 6])


from sklearn.impute import SimpleImputer
my_imputer = SimpleImputer(strategy = 'most_frequent')
imputed_X_test = pd.DataFrame(my_imputer.fit_transform(X_test))
imputed_X_test.columns = X_test.columns

onehotencoder = OneHotEncoder(categorical_features = [1])
imputed_X_test = onehotencoder.fit_transform(imputed_X_test).toarray()

onehotencoder = OneHotEncoder(categorical_features = [7])
imputed_X_test = onehotencoder.fit_transform(imputed_X_test).toarray()



# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
imputed_X_test = sc.transform(imputed_X_test)

# Fitting Naive Bayes to the Training set
from sklearn.svm import SVC
classifier = SVC(kernel = 'poly', random_state = 0)
classifier.fit(X_train, y_train)




# Predicting the Test set results
y_pred = classifier.predict(imputed_X_test)

datasetsub = pd.read_csv('gender_submission.csv')
datasetsub['Survived'] = y_pred
datasetsub.to_csv('data16.csv')


# Applying k-Fold Cross Validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)
accuracies.mean()
accuracies.std()

# Applying Grid Search to find the best model and the best parameters
from sklearn.model_selection import GridSearchCV
parameters = [{'bootstrap' : [True], 'max_depth' : [80,90,100,110], 'max_features' : [2,3], 'min_samples_leaf' : [3,4,5], 'min_samples_split' : [8,10,12]}]
grid_search = GridSearchCV(estimator = classifier,
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 3,
                           n_jobs = -1)
grid_search = grid_search.fit(X_train, y_train)
best_accuracy = grid_search.best_score_
best_parameters = grid_search.best_params_
