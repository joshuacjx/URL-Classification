import math
import pandas as pd
from parser import parse
from sklearn.metrics import f1_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV


INDENTATION = '  '


class NaiveBayesModel:
    def __init__(self, min_n, max_n):
        self.clf = MultinomialNB()
        self.count_vect = CountVectorizer(ngram_range=(min_n, max_n))

    def train(self, X_train, y_train):
        x_train_counts = self.count_vect.fit_transform(X_train)
        x_train_tfidf = TfidfTransformer().fit_transform(x_train_counts)
        self.clf.fit(x_train_tfidf, y_train)

    def predict(self, X_test):
        x_test_counts = self.count_vect.transform(X_test)
        x_test_tfidf = TfidfTransformer().fit_transform(x_test_counts)
        return self.clf.predict(x_test_tfidf)


# Read data
data = pd.read_csv('data/balanced_data.csv', header=None)
raw_X_data = data[0].tolist()
X_data = [' '.join(parse(url)) for url in raw_X_data]
y_data = data[1].tolist()
partitioning_ratios = (0.7, 0.2, 0.1)
last_train_idx = math.floor(partitioning_ratios[0]*len(X_data))
last_validation_idx = last_train_idx + math.floor(partitioning_ratios[1]*len(X_data))

model = NaiveBayesModel(min_n=1, max_n=4)

# Training
X_train = X_data[:last_validation_idx]
y_train = y_data[:last_validation_idx]
model.train(X_train, y_train)

# Tuning of hyper-parameters
# param_grid = dict(min_n=[1], max_n=[4])
# build_model = lambda min_n, max_n: NaiveBayesModel(min_n, max_n)
# classifier = KerasClassifier(build_fn=build_model, epochs=10)
# grid = GridSearchCV(estimator=classifier, param_grid=param_grid, cv=5)
# grid_result = grid.fit(X_train, y_train)
# print('Best: %f using %s' % (grid_result.best_score_, grid_result.best_params_))

# Validation
X_validation = X_data[last_train_idx+1:last_validation_idx]
y_validation_answer = y_data[last_train_idx+1:last_validation_idx]
y_validation_pred = model.predict(X_validation)
validation_score = f1_score(y_validation_answer, y_validation_pred, average='macro')
print(INDENTATION + 'Score on validation = {}'.format(validation_score))

# Testing
X_test = X_data[last_validation_idx+1:]
y_test_answer = y_data[last_validation_idx+1:]
y_test_pred = model.predict(X_test)
test_score = f1_score(y_test_answer, y_test_pred, average='macro')
print(INDENTATION + 'Score on testing = {}'.format(test_score))
