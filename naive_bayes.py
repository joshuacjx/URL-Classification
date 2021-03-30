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


def generate_result(test, y_pred, filename):
    test['Verdict'] = pd.Series(y_pred)
    test.drop(columns=['Text'], inplace=True)
    test.to_csv(filename, index=False)


def run_model(path, ngram_range, train_percentage):
    print("Running model on " + path)

    train = pd.read_csv(path)
    raw_X_data = train['URL'].tolist()
    X_data = [" ".join(parse(url)) for url in raw_X_data]
    y_data = train['Verdict'].tolist()

    num = len(X_data)
    last_train_idx = math.floor(train_percentage*num)

    X_train = X_data[:last_train_idx]
    y_train = y_data[:last_train_idx]

    model, count_vect = train_model(X_train, y_train, ngram_range)

    X_test = X_data[last_train_idx+1:]
    y_answer = y_data[last_train_idx+1:]
    y_pred = predict(model, count_vect, X_test)

    score = f1_score(y_answer, y_pred, average='macro')
    print('F-Score = {}'.format(score))


run_model("data/1_1_1_1_80000.csv", ngram_range=(1,6), train_percentage=0.8)
# run_model("data/1_5_5_1_60000.csv", ngram_range=(1,6), train_percentage=0.8)
# run_model("data/1_15_15_1_160000.csv", ngram_range=(1,6), train_percentage=0.8)
