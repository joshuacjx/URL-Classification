import math
import pandas as pd
from parser import parse
from sklearn.metrics import f1_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB


INDENTATION = '  '


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


def generate_result(test, y_pred, filename):
    test['Verdict'] = pd.Series(y_pred)
    test.drop(columns=['Text'], inplace=True)
    test.to_csv(filename, index=False)


def run_model(path, ngram_range, partitioning_ratios):
    print("Running Naive Bayes model on " + path)

    train = pd.read_csv(path)
    raw_X_data = train['URL'].tolist()
    X_data = [" ".join(parse(url)) for url in raw_X_data]
    y_data = train['Verdict'].tolist()

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


run_model("data/1_1_1_1_80000.csv", ngram_range=(1, 4), partitioning_ratios=(0.7, 0.2, 0.1))
run_model("data/1_5_5_1_60000.csv", ngram_range=(1, 4), partitioning_ratios=(0.7, 0.2, 0.1))
run_model("data/1_15_15_1_160000.csv", ngram_range=(1, 4), partitioning_ratios=(0.7, 0.2, 0.1))
