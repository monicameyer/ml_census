__author__ = 'monicameyer'

import pandas as pd
import numpy as np
import random
import sys
import tester
from chooseFeature import chooseFeature
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.preprocessing import OneHotEncoder
from sklearn import linear_model
from collections import Counter
import pickle


# Each of the next 6 functions take training and test data and run a model based on hyperparameters already
# chosen by our analysis. Then it dumps the model into a pickle object. It predicts the y_test values and
# returns those predictions.

def naiveBayes(X_train, y_train, X_test):
    kbest = SelectKBest(chi2, k=136)
    X_train = kbest.fit_transform(X_train, y_train)
    X_test = kbest.transform(X_test)
    gnb = GaussianNB()
    gnb.fit(X_train, y_train)
    # file = open("naiveBayesFit.pkl", "w")
    # pickle.dump(gnb, file)
    y_pred = gnb.predict(X_test)
    return y_pred


def chooseFeat(X_train, y_train, X_test):
    clf = chooseFeature()
    clf.fit(X_train, y_train)
    # file = open("chooseFeatFit.pkl", "w")
    # pickle.dump(clf, file)
    prediction = clf.predict(X_test)
    y_pred = np.asarray(prediction)
    return y_pred


def knn(X_train, y_train, X_test):
    knn = KNeighborsClassifier(n_neighbors=12, weights="distance")
    knn.fit(X_train, y_train)
    # file = open("knnFit.pkl", "w")
    # pickle.dump(knn, file)
    y_pred = knn.predict(X_test)
    return y_pred


def logistic(X_train, y_train, X_test):
    log = linear_model.LogisticRegression(penalty="l1", C=.25)
    log.fit(X_train, y_train)
    # file = open("logisticFit.pkl", "w")
    # pickle.dump(log, file)
    y_pred = log.predict(X_test)
    return y_pred


def randomForest(X_train, y_train, X_test):
    rf = RandomForestClassifier(n_estimators=20, max_features='log2')
    rf.fit(X_train, y_train)
    X_train_reduced = rf.transform(X_train)
    X_test_reduced = rf.transform(X_test)
    rf.fit(X_train_reduced, y_train)
    # file = open("randomForestFit.pkl", "w")
    # pickle.dump(rf, file)
    y_pred = rf.predict(X_test_reduced)
    return y_pred


def SVM(X_train, y_train, X_test):
    clf = SVC(C=3)
    clf.fit(X_train, y_train)
    # file = open("svmFit.pkl", "w")
    # pickle.dump(clf, file)
    y_pred = clf.predict(X_test)
    return y_pred


def sampling(X_train, y_train, rate):
    """
    This function takes the training data and produces balanced training data with the target
    containing equal numbers of positive and negative classes. It takes a rate so that we can choose
    the size of the training sample.
    """
    if rate > 1 and rate < 0.01:
        print "Rate must be between 0.01 and 1 (1% and 100%)"
        sys.exit()

    positive_indexes = [i for i, item in enumerate(y_train) if item == 1]
    negative_indexes = [i for i, item in enumerate(y_train) if item == 0]
    length_train = int(rate*X_train.shape[0]/2.0)

    positive_sample = [random.choice(positive_indexes) for i in xrange(length_train)]
    negative_sample = [random.choice(negative_indexes) for i in xrange(length_train)]
    Xnew = np.vstack((X_train[positive_sample], X_train[negative_sample]))

    ynew_pos = []
    ynew_neg = []
    for i in positive_sample:
        ynew_pos.append(y_train[i])
    for j in negative_sample:
        ynew_neg.append(y_train[j])

    y_train = np.hstack((ynew_pos, ynew_neg))
    return Xnew, y_train


def encode(census_training, census_test):
    """
    This function takes the training and testing data as pandas dataframes and drops the feature 'instance weight'.
    It uses the two functions in tester.py to transform the data into X_train, y_train, X_test and y_test. It
    sets the target feature in both training and testing data equal to 1 if '50000+' and 0 otherwise. Then it
    uses OneHotEncoder to take just the categorical features and turns them into dummy variables.
    """
    census_training = census_training.drop(['instance weight'], axis=1)
    census_test = census_test.drop(['instance weight'], axis=1)

    numeric_cols = ['age', 'wage per hour', 'capital gains', 'capital losses',
                    'divdends from stocks', 'num persons worked for employer',
                    'weeks worked in year']
    all_vars = census_training.columns.values
    categorical_col_indexes = [i for i, var in enumerate(all_vars)
                               if var not in numeric_cols and var != 'target']
    census_dictionary = tester.transform_csv(census_training, target_col='target')
    census_test_dictionary = tester.transform_csv(census_test, target_col='target')
    X_train, y_train = tester.transform_sklearn_dictionary(census_dictionary)
    X_test, y_test = tester.transform_sklearn_dictionary(census_test_dictionary)
    X_train = np.asarray(X_train)
    X_test = np.asarray(X_test)
    y_train = np.asarray([1 if y == '50000+.' else 0 for y in y_train], dtype=int)
    y_test = np.asarray([1 if y == '50000+.' else 0 for y in y_test], dtype=int)
    enc = OneHotEncoder(categorical_features=categorical_col_indexes)
    X_train = enc.fit_transform(X_train).toarray()
    X_test = enc.transform(X_test).toarray()
    return X_train, y_train, X_test, y_test


def scoring(y_test, y_pred):
    """
    This function gets counts of true positives, true negatives, false positives and false negatives.
    Then it prints a confusion matrix and the accuracy, precision, recall and f1
    """
    full_count = Counter()
    for i in range(len(y_pred)):
        if y_pred[i] == 1:
            if y_test[i] == 1:
                full_count['true_pos'] += 1
            else:
                full_count['false_pos'] += 1
        elif y_pred[i] == 0:
            if y_test[i] == 0:
                full_count['true_neg'] += 1
            else:
                full_count['false_neg'] += 1

    print "\t\t\t\t Actual:"
    print "\t\t\t\t    -   |  +"
    print "Predicted:    - | %d | %d" % (full_count['true_neg'], full_count['false_neg'])
    print "\t\t      + | %d | %d" % (full_count['false_pos'], full_count['true_pos'])

    accuracy = float((full_count['true_neg']+full_count['true_pos'])/float(y_test.shape[0]))
    precision = float(full_count['true_pos']/float(full_count['true_pos'] + full_count['false_pos']))
    recall = float(full_count['true_pos']/float(full_count['true_pos'] + full_count['false_neg']))
    f1_score = float(2*precision*recall)/float(precision + recall)
    print 'Accuracy | Precision | Recall | F1'
    print '%f | %f | %f | %f' % (accuracy, precision, recall, f1_score)

    return f1_score


def train(choice):
    # Import csv, fix categorical variables, converting to sklearn format
    census_training = pd.read_csv('censusImputedNumeric.csv', header=0, na_values="NA")
    census_test = pd.read_csv('censusTestImputedNumeric.csv', header=0, na_values="NA")
    X_train, y_train, X_test, y_test = encode(census_training, census_test)

    # Prints the name of the algorithm chosen, then takes a stratified sample the size of the rate given to
    # sampling * the size of the data set. Then runs the function associated with the algorithm chosen to get
    # predictions.
    print "Algorithm: " + choice
    if choice == 'Naive Bayes':
        X_train, y_train = sampling(X_train, y_train, 1)
        y_pred = naiveBayes(X_train, y_train, X_test)
    elif choice == 'Choose Feature':
        X_train, y_train = sampling(X_train, y_train, 1)
        y_pred = chooseFeat(X_train, y_train, X_test)
    elif choice == 'K Nearest Neighbors':
        X_train, y_train = sampling(X_train, y_train, 1)
        y_pred = knn(X_train, y_train, X_test)
    elif choice == 'Logistic Regression':
        X_train, y_train = sampling(X_train, y_train, 1)
        y_pred = logistic(X_train, y_train, X_test)
    elif choice == 'Random Forest':
        X_train, y_train = sampling(X_train, y_train, 1)
        y_pred = randomForest(X_train, y_train, X_test)
    elif choice == 'Support Vector Machine':
        X_train, y_train = sampling(X_train, y_train, .25)
        y_pred = SVM(X_train, y_train, X_test)
    else:
        print "Incorrect algorithm choice"
        print "Choose from: "
        print "Choose Feature, K Nearest Neighbors, Logistic Regression, Random Forest, or Support Vector Machine"
        sys.exit()

    # Scores the predictions based on the actual test data
    scoring(y_test, y_pred)

# To start training/testing, run the function train with the string argument of the model name
if __name__ == '__main__':
    train('K Nearest Neighbors')
    train('Naive Bayes')
    train('Logistic Regression')
    train('Choose Feature')
    train('Random Forest')
    train('Support Vector Machine')