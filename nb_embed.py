import math
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import f1_score
from sklearn.naive_bayes import MultinomialNB


"""
One simple technique that seems to work reasonably well for short texts 
(e.g., a sentence or a tweet) is to compute the vector for each word in 
the document, and then aggregate them using the coordinate-wise mean, min, or max.
Source: https://stats.stackexchange.com/questions/221715/apply-word-embeddings-to-
        entire-document-to-get-a-feature-vector
"""


class NaiveBayesEmbedModel:
    def __init__(self):
        self.multi_nb = MultinomialNB()
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

    def to_X_embed(self, X_data):
        X_each_word_embed = [[self.get_embed(word) for word in sentence]
                             for sentence in X_data]
        X_aggregate = [np.mean(vect_list, axis=0) for vect_list in X_each_word_embed]
        self.scaler.fit(X_aggregate)
        X_scaled = self.scaler.transform(X_aggregate)
        return X_scaled

    def train(self, X_train, y_train):
        X_train_embed = self.to_X_embed(X_train)
        self.multi_nb.fit(X_train_embed, y_train)

    def predict(self, X_test):
        X_test_embed = self.to_X_embed(X_test)
        return self.multi_nb.predict(X_test_embed)


INDENT = '  '

# Read URL data
data = pd.read_csv('data/balanced_parsed_data.csv', header=None)
X_data = data[0].tolist()
y_data = data[1].tolist()

part_ratio = (0.7, 0.2, 0.1)
last_train_idx = math.floor(part_ratio[0] * len(X_data))
last_valid_idx = last_train_idx + math.floor(part_ratio[1] * len(X_data))

model = NaiveBayesEmbedModel()

# Training
X_train = X_data[:last_valid_idx]
y_train = y_data[:last_valid_idx]
model.train(X_train, y_train)

# Validation
X_valid = X_data[last_train_idx + 1:last_valid_idx]
y_valid_ans = y_data[last_train_idx + 1:last_valid_idx]
y_valid_pred = model.predict(X_valid)
valid_score = f1_score(y_valid_ans, y_valid_pred, average='macro')
print(INDENT + 'Score on validation = {}'.format(valid_score))

# Testing
X_test = X_data[last_valid_idx + 1:]
y_test_ans = y_data[last_valid_idx + 1:]
y_test_pred = model.predict(X_test)
test_score = f1_score(y_test_ans, y_test_pred, average='macro')
print(INDENT + 'Score on testing = {}'.format(test_score))
