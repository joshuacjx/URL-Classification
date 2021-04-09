import math
import pandas as pd
from parser import parse
from sklearn.metrics import f1_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB


class NaiveBayesFreqModel:
    def __init__(self, min_n, max_n):
        self.multi_nb = MultinomialNB()
        self.vectorizer = CountVectorizer(ngram_range=(min_n, max_n))

    def train(self, X_train, y_train):
        X_train_counts = self.vectorizer.fit_transform(X_train)
        self.multi_nb.fit(X_train_counts, y_train)

    def predict(self, X_test):
        X_test_counts = self.vectorizer.transform(X_test)
        return self.multi_nb.predict(X_test_counts)


INDENT = '  '

# Read URL data
print("Reading data...")
data = pd.read_csv('data/balanced_data.csv', header=None)
X_data = [' '.join(parse(url)) for url in data[0].tolist()]
y_data = data[1].tolist()
part_ratio = (0.7, 0.2, 0.1)
last_train_idx = math.floor(part_ratio[0] * len(X_data))
last_valid_idx = last_train_idx + math.floor(part_ratio[1] * len(X_data))

model = NaiveBayesFreqModel(min_n=1, max_n=4)

# Training
print("Training in progress...")
X_train = X_data[:last_valid_idx]
y_train = y_data[:last_valid_idx]
model.train(X_train, y_train)

# Validation
print("Validation in progress...")
X_valid = X_data[last_train_idx + 1:last_valid_idx]
y_valid_ans = y_data[last_train_idx + 1:last_valid_idx]
y_valid_pred = model.predict(X_valid)
valid_score = f1_score(y_valid_ans, y_valid_pred, average='macro')
print('Score on validation = {}'.format(valid_score))

# Testing
print("Testing in progress...")
X_test = X_data[last_valid_idx + 1:]
y_test_ans = y_data[last_valid_idx + 1:]
y_test_pred = model.predict(X_test)
test_score = f1_score(y_test_ans, y_test_pred, average='macro')
print('Score on testing = {}'.format(test_score))


"""
Score on validation = 0.9519147465279099
Score on testing = 0.6845611687913808
"""
