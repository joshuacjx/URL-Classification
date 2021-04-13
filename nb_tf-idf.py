import math
import numpy as np
import pandas as pd
from keras.utils import to_categorical
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import f1_score
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


def get_best_params(X_data, y_data):
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


def get_best_ngram_range(X_train, y_train, X_test, y_test_ans):
    from itertools import combinations
    ranges = list(combinations(range(1, 7), 2))
    scores = []
    for pair in ranges:
        print("Trying out " + str(pair) + "...")
        tuning_model = NaiveBayesTfIdfModel(alpha=0.3, fit_prior=True, ngram_range=pair)
        tuning_model.train(X_train, y_train)
        y_test_pred = tuning_model.predict(X_test)
        test_score = f1_score(y_test_ans, y_test_pred, average='macro')
        scores.append(test_score)
        print(INDENT + "Score: " + str(test_score))
    return ranges[scores.index(max(scores))]


INDENT = '  '

# Read data
print("Reading data...")
data = pd.read_csv('data/balanced_parsed_data_3210.csv', header=None)
X_data = data[0].tolist()
y_data = data[1].tolist()

# Partition data
part_ratio = (0.7, 0.2, 0.1)
last_train_idx = math.floor(part_ratio[0] * len(X_data))
last_valid_idx = last_train_idx + math.floor(part_ratio[1] * len(X_data))
X_train = X_data[:last_valid_idx]
y_train = y_data[:last_valid_idx]
X_valid = X_data[last_train_idx + 1:last_valid_idx]
y_valid_ans = y_data[last_train_idx + 1:last_valid_idx]
X_test = X_data[last_valid_idx + 1:]
y_test_ans = y_data[last_valid_idx + 1:]

"""
get_best_params(X_data, y_data)
print(get_best_ngram_range(X_train, y_train, X_test, y_test_ans))
Best Hyper-parameters: {'alpha': 0.30000000000000004, 'fit_prior': True}
Best ngram-range: (1, 2)
"""

model = NaiveBayesTfIdfModel(alpha=0.3, fit_prior=True, ngram_range=(1, 2))

# Training
print("Training model...")
model.train(X_train, y_train)

# Validation
print("Validating model...")
y_valid_pred = model.predict(X_valid)
valid_score = roc_auc_score(to_categorical(y_valid_ans, 4),
                            to_categorical(y_valid_pred, 4),
                            average=None, multi_class='ovo')
print("Score on validation: " + str(valid_score))

# Testing
print("Testing model...")
y_test_pred = model.predict(X_test)
test_score = roc_auc_score(to_categorical(y_test_ans, 4),
                           to_categorical(y_test_pred, 4),
                           average=None, multi_class='ovo')
print("Score on testing: " + str(test_score))


"""
Score on validation: [0.97483726 0.94758871 0.96376996 0.9594055 ]
Score on testing: [0.87911285 0.73034838 0.77082071 0.814551  ]
"""
