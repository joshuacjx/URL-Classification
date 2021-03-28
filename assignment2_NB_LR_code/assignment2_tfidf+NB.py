#!/usr/bin/env python.

"""
CS4248 ASSIGNMENT 2 Template

TODO: Modify the variables below.  Add sufficient documentation to cross
reference your code with your writeup.

"""

# Import libraries.  Add any additional ones here.
# Generally, system libraries precede others.
import pandas as pd
import numpy as np
import torch
from collections import Counter
from sklearn.metrics import f1_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB

# TODO: Replace with your Student Number
_STUDENT_NUM = 'A0188119W'

def train_model(model, X_train, y_train):
    ''' TODO: train your model based on the training data '''
    count_vect = CountVectorizer(ngram_range=(5,8))
    x_train_counts = count_vect.fit_transform(X_train)
    x_train_tfidf = TfidfTransformer().fit_transform(x_train_counts)
    clf = MultinomialNB().fit(x_train_tfidf, y_train)
    return clf, count_vect
    
def predict(model, count_vect, X_test):
    ''' TODO: make your prediction here '''
    x_test_counts = count_vect.transform(X_test)
    x_test_tfidf = TfidfTransformer().fit_transform(x_test_counts)
    # print(x_train_tfidf.dimen)
    return model.predict(x_test_tfidf)

def generate_result(test, y_pred, filename):
    ''' generate csv file base on the y_pred '''
    test['Verdict'] = pd.Series(y_pred)
    test.drop(columns=['Text'], inplace=True)
    test.to_csv(filename, index=False)

def main():
    ''' load train, val, and test data '''
    train = pd.read_csv('train.csv')
    X_train = train['Text'].tolist()
    y_train = train['Verdict'].tolist()
    model = None # TODO: Define your model here

    model, count_vect = train_model(model, X_train, y_train)
    # test your model
    y_pred = predict(model, count_vect, X_train)

    # Use f1-macro as the metric
    score = f1_score(y_train, y_pred, average='macro')
    print('score on validation = {}'.format(score))

    # generate prediction on test data
    test = pd.read_csv('test.csv')
    X_test = test['Text'].tolist()
    y_pred = predict(model, count_vect, X_test)
    generate_result(test, y_pred, _STUDENT_NUM + ".csv")

# Allow the main class to be invoked if run as a file.
if __name__ == "__main__":
    main()
