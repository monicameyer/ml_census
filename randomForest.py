import tester
import pandas as pd
import numpy as np
import random
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.cross_validation import KFold
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

# Import training data
census_training = pd.read_csv('censusImputedNumeric.csv', header=0, na_values="NA")
census_training = census_training.drop(['instance weight'], axis=1)

# Identify categorical features that need to be converted to dummy variables
# by eliminating numerical features and target from
numeric_cols = ['age', 'wage per hour', 'capital gains', 'capital losses',
                'divdends from stocks', 'num persons worked for employer',
                'weeks worked in year']
all_vars = census_training.columns.values
categorical_col_indexes = [i for i, var in enumerate(all_vars)
                           if var not in numeric_cols and var != 'target']

# Create a dictionary of the data components that matches the structure of
# sklearn's built-in datasets
census_dictionary = tester.transform_csv(census_training, target_col='target')

# Extract the training predictors and target and convert into numpy arrays
X_train, y_train = tester.transform_sklearn_dictionary(census_dictionary)
X_train = np.asarray(X_train)
y_train = np.asarray([1 if y=='50000+.' else 0 for y in y_train], dtype=int)

# Convert the categorical features to multiple dummy variables; one dummy variable
# per category per feature
enc = OneHotEncoder(categorical_features=categorical_col_indexes)
X_train = enc.fit_transform(X_train).toarray()

# Resample the positive and negative indexes to create a 50-50 balanced sample
# of the training data
half = int(len(y_train)/2.0)
positive_indexes = [i for i, item in enumerate(y_train) if item==1]
negative_indexes = [i for i, item in enumerate(y_train) if item==0]
positive_sample = np.array([random.choice(positive_indexes) for i in xrange(half)])
negative_sample = np.array([random.choice(negative_indexes) for i in xrange(half)])
X_train_stratified = np.vstack((X_train[positive_sample], X_train[negative_sample]))
y_train_stratified = np.hstack((y_train[positive_sample], y_train[negative_sample]))

# Fit a random forest on the stratified/balanced data
clf = RandomForestClassifier()
clf.fit(X_train_stratified, y_train_stratified)

# Reduce the number of features by only taking features that have above average
# feature importance determined by random forest
X_train_stratified_reduced = clf.transform(X_train_stratified)
X_train_reduced = clf.transform(X_train)

# Find the best "primary" hyperparameters with a grid search over a specific
# parameter space
parameter_space = {'n_estimators': [10, 20, 50],
                   'max_features': ['auto', 'log2', .2, .4, .6]}
scores = ['precision', 'recall', 'accuracy', 'f1']

# Perform this grid search for multiple scoring metrics and print results
for score in scores:
    print "Tuning hyper-parameters for %s" % score
    clf = GridSearchCV(RandomForestClassifier(), parameter_space, cv=5, scoring=score)
    clf.fit(X_train_stratified_reduced, y_train_stratified)

    print "Best parameters set found on training set:"
    print clf.best_estimator_

    print "Grid scores on development set:"
    for params, mean_score, scores in clf.grid_scores_:
        print("%0.3f (+/-%0.03f) for %r"
              % (mean_score, scores.std() / 2, params))

# Define the new random forest classifier based on optimal model chosen above
clf = RandomForestClassifier(n_estimators=20, max_features='log2')

# Find the best "secondary" hyperparameters with a grid search
parameter_space = {'n_estimators': [20],
                   'max_features': ['log2'],
                   'criterion': ['gini', 'entropy'],
                   'min_samples_split': [2, 100, 1000, 2000],
                   'min_samples_leaf': [1, 50, 100, 1000]}
scores = ['precision', 'recall', 'accuracy', 'f1']

# Grid search for multiple scoring metrics
for score in scores:
    print "Tuning hyper-parameters for %s" % score
    clf = GridSearchCV(RandomForestClassifier(), parameter_space, cv=5, scoring=score)
    clf.fit(X_train_stratified_reduced, y_train_stratified)

    print "Best parameters set found on training set:"
    print clf.best_estimator_

    print "Grid scores on development set:"
    for params, mean_score, scores in clf.grid_scores_:
        print("%0.3f (+/-%0.03f) for %r"
              % (mean_score, scores.std() / 2, params))

# Best secondary hyperparameters were the sklearn defaults!
clf = RandomForestClassifier(n_estimators=20, max_features='log2')

# Perform 5-fold cross validation on the stratified, feature reduced training sample
# and print the scores on multiple metrics for each fold
folds = KFold(len(X_train_stratified_reduced), n_folds=5, shuffle=True)

print "f1    accuracy    precision    recall"

for train_index, test_index in folds:
    X_cv_train = X_train_stratified_reduced[train_index]
    X_cv_test = X_train_stratified_reduced[test_index]
    y_cv_train = y_train_stratified[train_index]
    y_cv_test = y_train_stratified[test_index]
    clf.fit(X_cv_train, y_cv_train)
    guess = clf.predict(X_cv_test)

    # Print the f1, recall, and precision with respect to the positive class
    print (f1_score(y_cv_test, guess, pos_label=1), accuracy_score(y_cv_test, guess),
            precision_score(y_cv_test, guess, pos_label=1), recall_score(y_cv_test, guess, pos_label=1))