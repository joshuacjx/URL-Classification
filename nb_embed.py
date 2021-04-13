import math
import numpy as np
import pandas as pd
from keras.utils import to_categorical
from sklearn.metrics import roc_auc_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import MinMaxScaler


class NaiveBayesEmbedModel:
    def __init__(self, alpha, fit_prior):
        self.multi_nb = MultinomialNB(alpha=alpha, fit_prior=fit_prior)
        self.embeds = dict()
        self.load_embeds()
        self.scaler = MinMaxScaler()

    def load_embeds(self):
        with open('data/glove.6B.50d.txt', 'r+', encoding="utf-8") as f:
            for line in f:
                parts = line.split()
                word = parts[0]
                embedding = np.asarray(parts[1:], dtype=np.float32)
                self.embeds[word] = embedding

    def get_embed(self, word):
        if word in self.embeds:
            return self.embeds[word]
        return self.embeds['<unk>']

    def to_X_embed_scaled(self, X_data):
        X_each_word_embed = [[self.get_embed(word) for word in sentence]
                             for sentence in X_data]
        X_aggregate = [np.mean(vect_list, axis=0) for vect_list in X_each_word_embed]
        self.scaler.fit(X_aggregate)
        X_scaled = self.scaler.transform(X_aggregate)
        return X_scaled

    def train(self, X_train, y_train):
        X_train_embed = self.to_X_embed_scaled(X_train)
        self.multi_nb.fit(X_train_embed, y_train)

    def predict(self, X_test):
        X_test_embed = self.to_X_embed_scaled(X_test)
        return self.multi_nb.predict(X_test_embed)


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
        roc_auc = roc_auc_score(new_actual_class, new_pred_class, average=average)
        roc_auc_dict[per_class] = roc_auc
    return roc_auc_dict


def get_best_params(X_data, y_data):
    from sklearn.model_selection import RepeatedStratifiedKFold
    from sklearn.model_selection import GridSearchCV
    tuning_model = MultinomialNB()
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    space = dict()
    space['alpha'] = list(np.arange(0.1, 2.1, 0.1))
    space['fit_prior'] = [True, False]
    search = GridSearchCV(tuning_model, space, scoring='accuracy', n_jobs=-1, cv=cv)
    X_train_embed = NaiveBayesEmbedModel(alpha=0.3, fit_prior=True).to_X_embed_scaled(X_data)
    result = search.fit(X_train_embed, y_data)
    print('Best Score: %s' % result.best_score_)
    print('Best Hyper-parameters: %s' % result.best_params_)


INDENT = '  '

# Read data
print("Reading data...")
data = pd.read_csv('data/balanced_parsed_data.csv', header=None)
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
Best Hyper-parameters: {'alpha': 0.30000000000000004, 'fit_prior': False}
"""

model = NaiveBayesEmbedModel(alpha=0.3, fit_prior=False)

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
