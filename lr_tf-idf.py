import math
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
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
        self.logistic_reg.fit(x_train_tfidf, y_train)

    def predict(self, X_test):
        x_train_counts = self.vectorizer.transform(X_test)
        x_train_tfidf = TfidfTransformer().fit_transform(x_train_counts).toarray()
        return self.logistic_reg.predict(x_train_tfidf)


def roc_auc_score_multiclass(actual_class, pred_class, average="macro"):
    """
        Source: https://stackoverflow.com/questions/39685740/
                calculate-sklearn-roc-auc-score-for-multi-class
    """
    unique_class = set(actual_class)
    roc_auc_dict = {}
    for per_class in unique_class:
        other_class = [x for x in unique_class if x != per_class]
        new_actual_class = [0 if x in other_class else 1 for x in actual_class]
        new_pred_class = [0 if x in other_class else 1 for x in pred_class]
        roc_auc = roc_auc_score(new_actual_class, new_pred_class, average = average)
        roc_auc_dict[per_class] = roc_auc
    return roc_auc_dict


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
valid_score = roc_auc_score_multiclass(y_valid_ans, y_valid_pred)
print("Score on validation: " + str(valid_score))

# Testing
print("Testing in progress...")
X_test = X_data[last_valid_idx + 1:]
y_test_ans = y_data[last_valid_idx + 1:]
y_test_pred = model.predict(X_test)
test_score = roc_auc_score_multiclass(y_test_ans, y_test_pred)
print("Score on testing: " + str(test_score))
