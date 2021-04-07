import math
import numpy as np
import pandas as pd

from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.metrics import f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from parser import parse


INDENTATION = '  '


class LemmaTokenizer(object):
    def __init__(self):
        super().__init__()
        self.lemmatizer = WordNetLemmatizer()

    def __call__(self, text):
        return [self.lemmatizer.lemmatize(t.lower()) for t in word_tokenize(text)]


def train_model(model, X_train, y_train):
    count_vect = CountVectorizer(ngram_range=(1, 2), max_df=0.75, min_df=5,
                                 max_features=10000, tokenizer=LemmaTokenizer())
    x_train_counts = count_vect.fit_transform(X_train)
    x_train_tfidf = TfidfTransformer().fit_transform(x_train_counts).toarray()
    all_features = np.array(list(list(tfidf) for tfidf in x_train_tfidf)).T.tolist()
    v = np.array(all_features).T
    print(INDENTATION + "Commence training...")
    clf = model.fit(v, y_train)
    print(INDENTATION + "Finished training!")
    return clf, count_vect


def predict(model, count_vect, X_test):
    x_train_counts = count_vect.transform(X_test)
    x_train_tfidf = TfidfTransformer().fit_transform(x_train_counts).toarray()
    all_features = np.array(list(list(tfidf) for tfidf in x_train_tfidf)).T.tolist()
    v = np.array(all_features).T
    return model.predict(v)


def run_model(path, partitioning_ratios):
    print("Running Logistic Regression model on " + path)

    train = pd.read_csv(path, header=None)
    raw_X_data = train[0].tolist()
    X_data = [" ".join(parse(url)) for url in raw_X_data]
    y_data = train[1].tolist()

    num = len(X_data)
    last_train_idx = math.floor(partitioning_ratios[0] * num)
    last_validation_idx = last_train_idx + math.floor(partitioning_ratios[1] * num)

    X_train = X_data[:last_validation_idx]
    y_train = y_data[:last_validation_idx]
    model = LogisticRegression(max_iter=5000, penalty='l1', solver='saga')
    model, count_vect = train_model(model, X_train, y_train)

    X_validation = X_data[last_train_idx+1:last_validation_idx]
    y_validation_answer = y_data[last_train_idx+1:last_validation_idx]
    y_validation_pred = predict(model, count_vect, X_validation)
    validation_score = f1_score(y_validation_answer, y_validation_pred, average='macro')
    print(INDENTATION + 'Score on validation = {}'.format(validation_score))

    X_test = X_data[last_validation_idx + 1:]
    y_test_answer = y_data[last_validation_idx + 1:]
    y_test_pred = predict(model, count_vect, X_test)
    test_score = f1_score(y_test_answer, y_test_pred, average='macro')
    print(INDENTATION + 'Score on testing = {}'.format(test_score))


run_model("data/balanced_data.csv", partitioning_ratios=(0.7, 0.2, 0.1))
