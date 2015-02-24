import tester
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.learning_curve import learning_curve
from sklearn.svm import SVC
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import scale
from sklearn.utils import shuffle
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

# SVM with default hyperparameters (C=1, kernel=rbf)
clf = SVC()

# Take a random 25% sample from the training data
sample_size = int(.25*len(X_train))
sample_index = np.array([random.choice(xrange(len(X_train))) for i in xrange(sample_size)])
X_sample = X_train[sample_index]
y_sample = y_train[sample_index]

# Perform 5-fold cross validation on 25% sample of training data and print results
folds = KFold(len(X_sample), n_folds=5, shuffle=True)

print "SVM on 25% random sample of training data\n"
print "f1    accuracy    precision    recall"
for train_index, test_index in folds:
    X_cv_train = X_sample[train_index]
    X_cv_test = X_sample[test_index]
    y_cv_train = y_sample[train_index]
    y_cv_test = y_sample[test_index]
    clf.fit(X_cv_train, y_cv_train)
    guess = clf.predict(X_cv_test)
    print (f1_score(y_cv_test, guess, pos_label=1), accuracy_score(y_cv_test, guess),
            precision_score(y_cv_test, guess, pos_label=1), recall_score(y_cv_test, guess, pos_label=1))

# Scale the training data
X_train_scaled = scale(X_train)

# Take a random 25% sample from the scaled training data
sample_size = int(.25*len(X_train_scaled))
sample_index = np.array([random.choice(xrange(len(X_train_scaled))) for i in xrange(sample_size)])
X_scaled_sample = X_train_scaled[sample_index]
y_sample = y_train[sample_index]

# Perform 5-fold cross validation on 25% sample of scaled data and print results
folds = KFold(len(X_scaled_sample), n_folds=5, shuffle=True)

print "SVM on 25% sample with scaling and class_weights\n"
print "f1    accuracy    precision    recall"
for train_index, test_index in folds:
    X_cv_train = X_scaled_sample[train_index]
    X_cv_test = X_scaled_sample[test_index]
    y_cv_train = y_sample[train_index]
    y_cv_test = y_sample[test_index]
    clf.fit(X_cv_train, y_cv_train)
    guess = clf.predict(X_cv_test)
    print (f1_score(y_cv_test, guess, pos_label=1), accuracy_score(y_cv_test, guess),
            precision_score(y_cv_test, guess, pos_label=1), recall_score(y_cv_test, guess, pos_label=1))

# Take a stratified random sample by sampling equally from the positive and negative indexes
sample_size = int(.25*len(X_train_scaled)/2)
positive_indexes = [i for i, item in enumerate(y_train) if item==1]
negative_indexes = [i for i, item in enumerate(y_train) if item==0]
positive_sample = np.array([random.choice(positive_indexes) for i in xrange(sample_size)])
negative_sample = np.array([random.choice(negative_indexes) for i in xrange(sample_size)])
X_train_scaled_stratified = np.vstack((X_train_scaled[positive_sample], X_train_scaled[negative_sample]))
y_train_stratified = np.hstack((y_train[positive_sample], y_train[negative_sample]))

# Shuffle the data, because of the way we pasted the strata together
X_train_scaled_stratified, y_train_stratified = shuffle(X_train_scaled_stratified, y_train_stratified)

# Perform 5-fold cross validation on the scaled, stratified sample
folds = KFold(len(X_train_scaled_stratified), n_folds=5, shuffle=True)

print "SVM on 25% stratified sample on scaled training data\n"
print "f1    accuracy    precision    recall"
for train_index, test_index in folds:
    X_cv_train = X_train_scaled_stratified[train_index]
    X_cv_test = X_train_scaled_stratified[test_index]
    y_cv_train = y_train_stratified[train_index]
    y_cv_test = y_train_stratified[test_index]
    clf.fit(X_cv_train, y_cv_train)
    guess = clf.predict(X_cv_test)
    print (f1_score(y_cv_test, guess, pos_label=1), accuracy_score(y_cv_test, guess),
            precision_score(y_cv_test, guess, pos_label=1), recall_score(y_cv_test, guess, pos_label=1))

# Perform grid search of multiple penalty parameter values and optimize based on multiple
# scoring metrics, then print results
parameter_space = {'C': [1, 2, 3, 5, 10]}
scores = ['precision', 'recall', 'accuracy', 'f1']

for score in scores:
    print "Tuning hyper-parameters for %s" % score
    clf = GridSearchCV(SVC(), parameter_space, cv=5, scoring=score)
    clf.fit(X_train_scaled_stratified, y_train_stratified)

    print "Best parameters set found on training set:"
    print clf.best_estimator_

    print "Grid scores on development set:"
    for params, mean_score, scores in clf.grid_scores_:
        print("%0.3f (+/-%0.03f) for %r"
              % (mean_score, scores.std() / 2, params))

# Best SVM classifier based on grid search
clf = SVC(C=3)

# Obtain 5-fold cross-validation scores for different sizes of training data
train = int((4.0/5.0)*len(y_train_stratified) - 1)
train_sizes, train_scores, valid_scores = learning_curve(
    clf, X_train_scaled_stratified, y_train_stratified, cv=5,
    train_sizes=[int(.01*train), int(.05*train), int(.1*train), int(.2*train)])

# Find the average accuracy for each value of C
train_scores_mean = np.mean(train_scores, axis=1)
valid_scores_mean = np.mean(valid_scores, axis=1)

# Print training vs validation scores
print "\nSVC on Stratified, Scaled Census Data"
print " "*26 + "         ".join(['1%', '5%', '10%', '20%'])
print "Training Scores    " + str(train_scores_mean)
print "Validation Scores  " + str(valid_scores_mean) + '\n'

# Plot the learning curve
plt.title("Learning Curve for SVC on Census Data")
plt.xlabel("Training Size")
plt.ylabel("Score")
plt.ylim(0.0, 1.1)
plt.plot(xrange(len(train_sizes)), train_scores_mean, 'o-', label="Training score", color="r")
plt.plot(xrange(len(train_sizes)), valid_scores_mean, 'o-', label="Validation score", color="g")
plt.xticks(xrange(len(train_sizes)), train_sizes)
plt.legend(loc="best")
plt.savefig("learning_curve_svc.png", format='png')
plt.close()

