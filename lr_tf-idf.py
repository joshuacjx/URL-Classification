import math
import pandas as pd
from keras.utils import to_categorical
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer


class LogisticRegressionTfIdfModel:
    def __init__(self, C, penalty, solver, ngram_range):
        self.logistic_reg = LogisticRegression(max_iter=5000, penalty=penalty, solver=solver, C=C)
        self.vectorizer = CountVectorizer(ngram_range=ngram_range, max_df=0.75, min_df=5, max_features=10000)
        self.transformer = TfidfTransformer()

    def train(self, X_train, y_train):
        x_train_counts = self.vectorizer.fit_transform(X_train)
        x_train_tfidf = self.transformer.fit_transform(x_train_counts).toarray()
        self.logistic_reg.fit(x_train_tfidf, y_train)

    def predict(self, X_test):
        x_train_counts = self.vectorizer.transform(X_test)
        x_train_tfidf = TfidfTransformer().fit_transform(x_train_counts).toarray()
        return self.logistic_reg.predict(x_train_tfidf)


def get_best_params(X_data, y_data):
    from sklearn.model_selection import RepeatedStratifiedKFold
    from sklearn.model_selection import GridSearchCV
    tuning_model = LogisticRegression()
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    space = dict()
    space['solver'] = ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']
    space['penalty'] = ['none', 'l1', 'l2', 'elasticnet']
    space['C'] = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100]
    search = GridSearchCV(tuning_model, space, scoring='accuracy', n_jobs=-1, cv=cv)
    X_data_tfidf = TfidfTransformer().fit_transform(
        CountVectorizer(ngram_range=(1, 4)).fit_transform(X_data))
    result = search.fit(X_data_tfidf, y_data)
    print('Best Score: %s' % result.best_score_)
    print('Best Hyper-parameters: %s' % result.best_params_)


INDENT = '  '

# Read data
print("Reading data...")
MAX_DATA = 10000
data = pd.read_csv('data/balanced_parsed_data_3210.csv', header=None)
X_data = data[0].tolist()[:MAX_DATA]
y_data = data[1].tolist()[:MAX_DATA]

"""
# Parameter Tuning
get_best_params(X_data, y_data)
Best Score: 0.6106666666666668
Best Hyper-parameters: {'C': 100, 'penalty': 'l1', 'solver': 'saga'}
"""

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

model = LogisticRegressionTfIdfModel(C=100, penalty='l1', solver='saga', ngram_range=(1, 2))

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
Score on validation: [0.88632221 0.78718169 0.80864246 0.85150966]
Score on testing: [0.75636234 0.63769476 0.66917075 0.72316017]
"""
