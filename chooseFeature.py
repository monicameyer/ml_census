__author__ = 'monicameyer'

import sklearn
import pandas as pd
import numpy as np
import tester
import random
from collections import Counter
from sklearn import cross_validation
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.preprocessing import OneHotEncoder

class chooseFeature(sklearn.base.BaseEstimator):
    """
    This defines a classifier that predicts on the basis of
    the feature that was found to have the best weighted purity, based on splitting all
    features according to their mean value. Then, for that feature split, it predicts
    a new example based on the mean value of the chosen feature, and the majority class for
    that split.
    """

    def __init__(self):
        # if we haven't been trained, always return 1
        self.classForGreater = 1
        self.classForLeq = 1
        self.chosenFeature = 0
        self.type = "chooseFeatureClf"

    def impurity(self, labels):

        labelLength = float(len(labels))
        counter = Counter(labels)
        for item in counter:
            counter[item] = counter[item]/labelLength

        running_sum = 0
        for item in counter:
            square = (counter[item])**2
            running_sum += square
        index = 1-running_sum
        return index


    def weighted_impurity(self, list_of_label_lists):
        list_len = 0
        for list in list_of_label_lists:
            list_len += len(list)

        impurities = []
        for item in list_of_label_lists:
            length = len(item)

            impurities.append(self.impurity(item)*length/list_len)
        impurity = sum(impurities)
        return impurity


    def ftr_seln(self, data, labels):
        """return: index of feature with best weighted_impurity, when split
        according to its mean value; you are permitted to return other values as well,
        as long as the the first value is the index
        """
        num_features = len(data[0])

        impurities = []
        for i in range(num_features):
            column = data[:,i]
            mu = np.mean(column)
            less = []
            more = []
            for i in range(len(column)):
                if column[i] > mu:
                    more.append(labels[i])
                elif column[i] <= mu:
                    less.append(labels[i])
            impurities.append(self.weighted_impurity([less, more]))
        choice = impurities.index(min(impurities))
        return choice


    def fit(self, data, labels):

        self.chosenFeature = self.ftr_seln(data, labels)
        x = data[:,self.chosenFeature]

        mu = np.mean(x)
        self.mean = mu
        less = []
        more = []
        for i in range(len(x)):
            if x[i] > mu:
                more.append(labels[i])
            elif x[i] <= mu:
                less.append(labels[i])

        countsOver = Counter()
        countsUnder = Counter()

        try:
            for j in more:
                countsOver[j] += 1
            self.classForGreater = countsOver.most_common()[0][0]
            for k in less:
                countsUnder[k] += 1
            self.classForLeq = countsUnder.most_common()[0][0]
        except:pass


    def predict(self, testData):
        """
        Input: testData: a list of X vectors to label. Check the chosen feature of each
        element of testData and make a classification decision based on it
        """
        mu = self.mean
        x = testData[:,self.chosenFeature]

        result = []
        for i in range(len(x)):
            if x[i] > mu:
                result.append(self.classForGreater)
            elif x[i] <= mu:
                result.append(self.classForLeq)
        return result


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



def train():
    # Read in csv files into pandas dataframes
    census_training = pd.read_csv('censusImputedNumeric.csv', header=0, na_values="NA")
    census_test = pd.read_csv('censusTestImputedNumeric.csv', header=0, na_values="NA")

    # Encode both training and testing data, get a sample
    X_train, y_train, X_test, y_test = encode(census_training, census_test)
    X_, y_ = sampling(X_train, y_train, 1)

    # This instantiates an object of class chooseFeature, then runs through
    # cross validation of the model with 5 folds, scoring by f1.
    clf = chooseFeature()
    kf = cross_validation.KFold(len(X_), n_folds=5)
    f1 = []
    precision = []
    accuracy = []
    recall = []

    print "f1    accuracy    precision    recall"
    for train_idx, test_idx in kf:
        X_cv_train, X_cv_test = X_[train_idx], X_[test_idx]
        y_cv_train, y_cv_test = y_[train_idx], y_[test_idx]
        clf.fit(X_cv_train, y_cv_train)
        guess = clf.predict(X_cv_test)
        f1.append(f1_score(y_cv_test, guess, pos_label=1))
        accuracy.append(accuracy_score(y_cv_test, guess))
        precision.append(precision_score(y_cv_test, guess, pos_label=1))
        recall.append(recall_score(y_cv_test, guess, pos_label=1))

    # Gets the average f1 score of the cross validation and prints it.
    avg_f1 = np.mean(f1)
    avg_precision = np.mean(precision)
    print type(avg_precision)
    avg_accuracy = np.mean(accuracy)
    avg_recall = np.mean(recall)
    print "Averages:'"
    print "f1: %f, accuracy: %f, precision: %f, recall: %f" % (avg_f1, avg_accuracy, avg_precision, avg_recall)


if __name__ == '__main__':
    train()