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


def run_model(path, partitioning_ratios, max_data):
    print("Running Logistic Regression model on " + path)

    train = pd.read_csv(path)
    raw_X_data = train['URL'].tolist()[:max_data]
    X_data = [" ".join(parse(url)) for url in raw_X_data]
    y_data = train['Verdict'].tolist()[:max_data]

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


MAX_DATA = 10000
run_model("data/1_1_1_1_80000.csv", partitioning_ratios=(0.7, 0.2, 0.1), max_data=MAX_DATA)
run_model("data/1_5_5_1_60000.csv", partitioning_ratios=(0.7, 0.2, 0.1), max_data=MAX_DATA)
run_model("data/1_15_15_1_160000.csv", partitioning_ratios=(0.7, 0.2, 0.1), max_data=MAX_DATA)


"""
Output:
Running Logistic Regression model on data/1_1_1_1_80000.csv
  Commence training...
  Finished training!
  Score on validation = 0.6811064582382795
  Score on testing = 0.6101794671047824
Running Logistic Regression model on data/1_5_5_1_60000.csv
  Commence training...
  Finished training!
  Score on validation = 0.6050786274730141
  Score on testing = 0.5053070133531989
Running Logistic Regression model on data/1_15_15_1_160000.csv
  Commence training...
  Finished training!
  Score on validation = 0.5537133018891178
  Score on testing = 0.44066990555754604
"""
