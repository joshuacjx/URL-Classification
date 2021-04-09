import math
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import roc_auc_score
from sklearn.naive_bayes import MultinomialNB


class NaiveBayesTfIdfModel:
    def __init__(self, alpha, fit_prior, ngram_range):
        self.multi_nb = MultinomialNB(alpha=alpha, fit_prior=fit_prior)
        self.vectorizer = CountVectorizer(ngram_range=ngram_range)
        self.transformer = TfidfTransformer()

    def train(self, X_train, y_train):
        X_train_counts = self.vectorizer.fit_transform(X_train)
        X_train_tfidf = self.transformer.fit_transform(X_train_counts)
        self.multi_nb.fit(X_train_tfidf, y_train)

    def predict(self, X_test):
        X_test_counts = self.vectorizer.transform(X_test)
        X_test_tfidf = TfidfTransformer().fit_transform(X_test_counts)
        return self.multi_nb.predict(X_test_tfidf)


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


INDENT = '  '

# Read URL data
print("Reading data...")
data = pd.read_csv('data/balanced_parsed_data.csv', header=None)
X_data = data[0].tolist()
y_data = data[1].tolist()
part_ratio = (0.7, 0.2, 0.1)
last_train_idx = math.floor(part_ratio[0] * len(X_data))
last_valid_idx = last_train_idx + math.floor(part_ratio[1] * len(X_data))


def get_best_params():
    from sklearn.model_selection import RepeatedStratifiedKFold
    from sklearn.model_selection import GridSearchCV
    tuning_model = MultinomialNB()
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    space = dict()
    space['alpha'] = list(np.arange(0.1, 2.1, 0.1))
    space['fit_prior'] = [True, False]
    search = GridSearchCV(tuning_model, space, scoring='accuracy', n_jobs=-1, cv=cv)
    X_data_tfidf = TfidfTransformer().fit_transform(
        CountVectorizer(ngram_range=(1, 4)).fit_transform(X_data))
    result = search.fit(X_data_tfidf, y_data)
    print('Best Score: %s' % result.best_score_)
    print('Best Hyper-parameters: %s' % result.best_params_)


"""
get_best_params()
Best Hyper-parameters: {'alpha': 0.30000000000000004, 'fit_prior': True}
"""

model = NaiveBayesTfIdfModel(alpha=0.3, fit_prior=True, ngram_range=(1, 4))

# Training
print("Training model...")
X_train = X_data[:last_valid_idx]
y_train = y_data[:last_valid_idx]
model.train(X_train, y_train)

# Validation
print("Validating model...")
X_valid = X_data[last_train_idx + 1:last_valid_idx]
y_valid_ans = y_data[last_train_idx + 1:last_valid_idx]
y_valid_pred = model.predict(X_valid)
valid_score = roc_auc_score_multiclass(y_valid_ans, y_valid_pred)
print("Score on validation: " + str(valid_score))

# Testing
print("Testing model...")
X_test = X_data[last_valid_idx + 1:]
y_test_ans = y_data[last_valid_idx + 1:]
y_test_pred = model.predict(X_test)
test_score = roc_auc_score_multiclass(y_test_ans, y_test_pred)
print("Score on testing: " + str(test_score))
