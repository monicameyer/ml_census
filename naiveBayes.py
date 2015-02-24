__author__ = 'monicameyer'

import pandas as pd
import numpy as np
import random
import tester
from sklearn.naive_bayes import GaussianNB
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.cross_validation import KFold
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from collections import Counter


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
    X_test = enc.fit_transform(X_test).toarray()
    return X_train, y_train, X_test, y_test


def sampling(X_train, y_train, rate):
    """
    This function takes the training data and produces balanced training data with the target
    containing equal numbers of positive and negative classes. It takes a rate so that we can choose
    the size of the training sample.
    """
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


def pick_K():
    """
    This function runs through k = (2:489), running SelectKBest for each k to determine which number
    of k gives the best f1_score. This was tested changing the rate in the sampling function to .1, .2
    and 1 to run the model on 10%, 20% and 100% sample sizes of the data.
    """
    # Read in csv files into pandas dataframes
    census_training = pd.read_csv('censusImputedNumeric.csv', header=0, na_values="NA")
    census_test = pd.read_csv('censusTestImputedNumeric.csv', header=0, na_values="NA")
    # Encode both training and testing data
    X_train, y_train, X_test, y_test = encode(census_training, census_test)


    scores = Counter()
    for j in range(2,len(X_train[0])):
        print j
        X_ = SelectKBest(chi2, k=j).fit_transform(X_train, y_train)
        X_, y_ = sampling(X_, y_train, 1)
        gnb = GaussianNB()
        folds = KFold(len(X_), n_folds=2, shuffle=True)

        f1 = []
        for train_index, test_index in folds:
            X_cv_train, X_cv_test = X_[train_index], X_[test_index]
            y_cv_train, y_cv_test = y_[train_index], y_[test_index]
            gnb.fit(X_cv_train, y_cv_train)
            guess = gnb.predict(X_cv_test)
            f1.append(f1_score(y_cv_test, guess, pos_label=1))

        avg_f1 = np.mean(f1)
        scores[j] = avg_f1
    print scores.most_common(10)
    return

def train():
    # Read in csv files into pandas dataframes
    census_training = pd.read_csv('censusImputedNumeric.csv', header=0, na_values="NA")
    census_test = pd.read_csv('censusTestImputedNumeric.csv', header=0, na_values="NA")
    # Encode both training and testing data
    X_train, y_train, X_test, y_test = encode(census_training, census_test)
    X_ = SelectKBest(chi2, k=136).fit_transform(X_train, y_train)
    X_, y_ = sampling(X_, y_train, 1)
    gnb = GaussianNB()
    folds = KFold(len(X_), n_folds=2, shuffle=True)

    f1 = []
    precision = []
    accuracy = []
    recall = []
    print "f1    accuracy    precision    recall"
    for train_index, test_index in folds:
        X_cv_train, X_cv_test = X_[train_index], X_[test_index]
        y_cv_train, y_cv_test = y_[train_index], y_[test_index]
        gnb.fit(X_cv_train, y_cv_train)
        guess = gnb.predict(X_cv_test)
        f1.append(f1_score(y_cv_test, guess, pos_label=1))
        accuracy.append(accuracy_score(y_cv_test, guess))
        precision.append(precision_score(y_cv_test, guess, pos_label=1))
        recall.append(recall_score(y_cv_test, guess, pos_label=1))

    avg_f1 = np.mean(f1)
    avg_precision = np.mean(precision)
    avg_accuracy = np.mean(accuracy)
    avg_recall = np.mean(recall)
    print "Averages:'"
    print "f1: %f, accuracy: %f, precision: %f, recall: %f" % (avg_f1, avg_accuracy, avg_precision, avg_recall)
    return

if __name__ == '__main__':
    train()
    #pick_K()

