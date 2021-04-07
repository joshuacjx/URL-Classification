import math
import pandas as pd
from parser import parse
from sklearn.metrics import f1_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB


def train_model(X_train, y_train, ngram_range):
    count_vect = CountVectorizer(ngram_range=ngram_range)
    x_train_counts = count_vect.fit_transform(X_train)
    x_train_tfidf = TfidfTransformer().fit_transform(x_train_counts)
    clf = MultinomialNB().fit(x_train_tfidf, y_train)
    return clf, count_vect


def predict(model, count_vect, X_test):
    x_test_counts = count_vect.transform(X_test)
    x_test_tfidf = TfidfTransformer().fit_transform(x_test_counts)
    return model.predict(x_test_tfidf)


def run_model(path, ngram_range, partitioning_ratios):
    print("Running Naive Bayes model on " + path)
    INDENTATION = '  '

    train = pd.read_csv(path, header=None)
    raw_X_data = train[0].tolist()
    X_data = [" ".join(parse(url)) for url in raw_X_data]
    y_data = train[1].tolist()

    num = len(X_data)
    last_train_idx = math.floor(partitioning_ratios[0]*num)
    last_validation_idx = last_train_idx + math.floor(partitioning_ratios[1]*num)

    X_train = X_data[:last_validation_idx]
    y_train = y_data[:last_validation_idx]
    model, count_vect = train_model(X_train, y_train, ngram_range)

    X_validation = X_data[last_train_idx+1:last_validation_idx]
    y_validation_answer = y_data[last_train_idx+1:last_validation_idx]
    y_validation_pred = predict(model, count_vect, X_validation)
    validation_score = f1_score(y_validation_answer, y_validation_pred, average='macro')
    print(INDENTATION + 'Score on validation = {}'.format(validation_score))

    X_test = X_data[last_validation_idx+1:]
    y_test_answer = y_data[last_validation_idx+1:]
    y_test_pred = predict(model, count_vect, X_test)
    test_score = f1_score(y_test_answer, y_test_pred, average='macro')
    print(INDENTATION + 'Score on testing = {}'.format(test_score))


run_model("data/balanced_data.csv", ngram_range=(1, 4), partitioning_ratios=(0.7, 0.2, 0.1))


"""
Output:
Running Naive Bayes model on data/balanced_data.csv
  Score on validation = 0.9308420928127439
  Score on testing = 0.6775851446460903
"""
