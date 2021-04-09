import math
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer


INDENT = '  '


class LogisticRegressionTfIdfModel:
    def __init__(self, min_n, max_n, penalty):
        self.logistic_reg = LogisticRegression(max_iter=5000, penalty=penalty, solver='saga')
        self.vectorizer = CountVectorizer(ngram_range=(min_n, max_n), max_df=0.75, min_df=5, max_features=10000)
        self.transformer = TfidfTransformer()

    def train(self, X_train, y_train):
        x_train_counts = self.vectorizer.fit_transform(X_train)
        x_train_tfidf = self.transformer.fit_transform(x_train_counts).toarray()
        print(INDENT + 'Fitting model...')
        self.logistic_reg.fit(x_train_tfidf, y_train)

    def predict(self, X_test):
        x_train_counts = self.vectorizer.transform(X_test)
        x_train_tfidf = TfidfTransformer().fit_transform(x_train_counts).toarray()
        return self.logistic_reg.predict(x_train_tfidf)


# Read URL data
print("Reading data...")
MAX_DATA = 10000
data = pd.read_csv('data/balanced_parsed_data.csv', header=None)
X_data = data[0].tolist()[:MAX_DATA]
y_data = data[1].tolist()[:MAX_DATA]
part_ratio = (0.7, 0.2, 0.1)
last_train_idx = math.floor(part_ratio[0] * len(X_data))
last_valid_idx = last_train_idx + math.floor(part_ratio[1] * len(X_data))

model = LogisticRegressionTfIdfModel(min_n=1, max_n=1, penalty='l1')

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
print(INDENT + 'Score on validation = {}'.format(valid_score))

# Testing
print("Testing in progress...")
X_test = X_data[last_valid_idx + 1:]
y_test_ans = y_data[last_valid_idx + 1:]
y_test_pred = model.predict(X_test)
test_score = f1_score(y_test_ans, y_test_pred, average='macro')
print(INDENT + 'Score on testing = {}'.format(test_score))


"""
Score on validation = 0.6346830491700619
Score on testing = 0.6046870171381509
"""
