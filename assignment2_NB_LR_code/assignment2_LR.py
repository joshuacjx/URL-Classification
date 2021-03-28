#!/usr/bin/env python.

"""
CS4248 ASSIGNMENT 2 Template

TODO: Modify the variables below.  Add sufficient documentation to cross
reference your code with your writeup.

"""

# Import libraries.  Add any additional ones here.
# Generally, system libraries precede others.
import pandas as pd
from sklearn.metrics import f1_score
import numpy as np
import spacy
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer 
import re

OPNION_WORD = ["\\bthink", "\\bthought","\\bbelieve", "\\bagree", "\\bhave to", "\\bhas to", "\\bhad to", "\\bgot to", "\\bsay", "\\bsaid", "\\bsuggest", "\\bought to", "\\bnotice", "\\bsee", "\\btell", "\\bmean", "\\bdoubt"]

class LemmaTokenizer(object):
    def __init__(self):
        super().__init__()
        self.lemmatizer = WordNetLemmatizer() 

    def __call__(self, text):
        return [self.lemmatizer.lemmatize(t.lower()) for t in word_tokenize(text)]

# TODO: Replace with your Student Number
_STUDENT_NUM = 'A0188119W'

def getEntLabel(doc, label):
    for item in doc.ents:
        if item.label_ == label:
            return True
    return False

def getEntLabelList(l, label):
    res = [0] * len(l)
    for i in range(len(l)):
        if getEntLabel(l[i], label):
            res[i] = 1
    return res

def getPOSTag(doc, tag):
    for token in doc:
        if token.tag_ == tag:
            return True
    return False

def getPOSTagList(l, tag):
    res = [0] * len(l)
    for i in range(len(l)):
        if getPOSTag(l[i], tag):
            res[i] = 1
    return res

def train_model(model, nlp, features, X_train, y_train):
    labels = features[["Person-tag", "GPE-tag", "Law-tag", "Number-tag", 'Ordinal-tag', 'Cardinal-tag', 'Past-tense', 'Past-participle']].to_numpy()
    count_vect = CountVectorizer(ngram_range=(1,2),max_df=0.75,min_df=5,max_features=10000,tokenizer=LemmaTokenizer())
    x_train_counts = count_vect.fit_transform(X_train)
    x_train_tfidf = TfidfTransformer().fit_transform(x_train_counts).toarray()
    v = np.append(labels, x_train_tfidf, axis=1)
    print("start training")
    clf = model.fit(v, y_train)
    print("finish training")
    return clf, count_vect

def predict(model, nlp, features, count_vect, X_test):
    print("start predicting")
    labels = features[["Person-tag", "GPE-tag", "Law-tag", "Number-tag", 'Ordinal-tag', 'Cardinal-tag', 'Past-tense', 'Past-participle']].to_numpy()
    x_train_counts = count_vect.transform(X_test)
    x_train_tfidf = TfidfTransformer().fit_transform(x_train_counts).toarray()
    v = np.append(labels, x_train_tfidf, axis=1)
    return model.predict(v)

def generate_result(test, y_pred, filename):
    ''' generate csv file base on the y_pred '''
    test['Verdict'] = pd.Series(y_pred)
    test.drop(columns=['Text'], inplace=True)
    test.to_csv(filename, index=False)

def compute_features(nlp, writer, X_data, filename):
    print("getting nlp for texts")
    X_data_nlp = [nlp(text) for text in X_data]
    
    print("computing Named Entities")
    labels = ['PERSON', 'GPE', 'LAW', 'QUANTITY', 'ORDINAL', 'CARDINAL', 'ORG', 'PERCENT', 'NORP']
    columm_names_1 = ['Person-tag','GPE-tag','Law-tag', 'Quantity-tag', 'Ordinal-tag', 'Cardinal-tag', 'Org-tag', 'Percent-tag', 'NORP-tag']
    for i in range(len(labels)):
        writer[columm_names_1[i]] = getEntLabelList(X_data_nlp, labels[i])
    
    print("computing POS tags")
    tags = ['CD', 'VBD', 'VBN']
    column_names_2 = ['Number-tag', 'Past-tense', 'Past-participle']
    for i in range(len(tags)):
        writer[column_names_2[i]] = getPOSTagList(X_data_nlp, tags[i])
    
    writer['length'] = [len(x) for x in X_data]
    
    print("computing opnion words")
    contains_opnion = [0] * len(X_data)
    for i in range(len(X_data)):
        for r in OPNION_WORD:
            if re.search(r, X_data[i]):
                contains_opnion[i] = 1
                break
    writer['opinion-words'] = contains_opnion
    writer.to_csv(filename, index=False)

def main():
    ''' load train, val, and test data '''
    train = pd.read_csv('train.csv')
    X_train = train['Text'].tolist()
    y_train = train['Verdict'].tolist()
    nlp = spacy.load("en_core_web_sm")
    features = pd.read_csv("A0188119W_train_feature.csv")
    
    # compute features for training set
    #compute_features(nlp, features, X_train, _STUDENT_NUM + "_train_feature.csv")
    
    # train the model
    model = LogisticRegression(max_iter=500,penalty='l1',solver='saga') # TODO: Define your model here

    model, count_vect= train_model(model, nlp, features, X_train, y_train)
    # test your model
    y_pred = predict(model, nlp, features, count_vect, X_train)

    # Use f1-macro as the metric
    score = f1_score(y_train, y_pred, average='macro')
    print('score on validation = {}'.format(score))

    test = pd.read_csv('test.csv')
    X_test = test['Text'].tolist()
    test_features = pd.read_csv('A0188119W_test_feature.csv')
    
    # compute features for test set
    compute_features(nlp, test_features, X_test, _STUDENT_NUM + "_test_feature.csv")
    
    # generate prediction on test data
    y_pred = predict(model, nlp, test_features, count_vect, X_test)
    generate_result(test, y_pred, _STUDENT_NUM + ".csv")

# Allow the main class to be invoked if run as a file.
if __name__ == "__main__":
    main()
