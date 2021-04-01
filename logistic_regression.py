import pandas as pd
from sklearn.metrics import f1_score
import numpy as np
import spacy
import math
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import LogisticRegression
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
import re


class LemmaTokenizer(object):
    def __init__(self):
        super().__init__()
        self.lemmatizer = WordNetLemmatizer()

    def __call__(self, text):
        return [self.lemmatizer.lemmatize(t.lower()) for t in word_tokenize(text)]


INDENTATION = '  '

SELECTED_FEATURES = ["Person-tag", "GPE-tag", "Law-tag",
                     "Number-tag", 'Ordinal-tag', 'Cardinal-tag',
                     'Past-tense', 'Past-participle']


def compute_features(nlp, writer, X_data, filename):
    OPNION_WORD = ["\\bthink", "\\bthought", "\\bbelieve", "\\bagree", "\\bhave to",
                   "\\bhas to", "\\bhad to", "\\bgot to", "\\bsay", "\\bsaid", "\\bsuggest",
                   "\\bought to", "\\bnotice", "\\bsee", "\\btell", "\\bmean", "\\bdoubt"]

    def getEntLabel(doc, label):
        for item in doc.ents:
            if item.label_ == label:
                return True
        return False

    def getEntLabelList(l, label):
        res = [0] * len(l)
        for i in range(len(l)):
            if getEntLabel(l[i], label):
                res[i] = 1
        return res

    def getPOSTag(doc, tag):
        for token in doc:
            if token.tag_ == tag:
                return True
        return False

    def getPOSTagList(l, tag):
        res = [0] * len(l)
        for i in range(len(l)):
            if getPOSTag(l[i], tag):
                res[i] = 1
        return res

    print("getting nlp for texts")
    X_data_nlp = [nlp(text) for text in X_data]

    print("computing Named Entities")
    labels = ['PERSON', 'GPE', 'LAW', 'QUANTITY', 'ORDINAL',
              'CARDINAL', 'ORG', 'PERCENT', 'NORP']
    columm_names_1 = ['Person-tag', 'GPE-tag', 'Law-tag',
                      'Quantity-tag', 'Ordinal-tag', 'Cardinal-tag',
                      'Org-tag', 'Percent-tag', 'NORP-tag']
    for i in range(len(labels)):
        writer[columm_names_1[i]] = getEntLabelList(X_data_nlp, labels[i])

    print("computing POS tags")
    tags = ['CD', 'VBD', 'VBN']
    column_names_2 = ['Number-tag', 'Past-tense', 'Past-participle']
    for i in range(len(tags)):
        writer[column_names_2[i]] = getPOSTagList(X_data_nlp, tags[i])

    writer['length'] = [len(x) for x in X_data]

    print("computing opnion words")
    contains_opnion = [0] * len(X_data)
    for i in range(len(X_data)):
        for r in OPNION_WORD:
            if re.search(r, X_data[i]):
                contains_opnion[i] = 1
                break
    writer['opinion-words'] = contains_opnion
    writer.to_csv(filename, index=False)


def train_model(model, nlp, features, X_train, y_train):
    labels = features[SELECTED_FEATURES].to_numpy()
    count_vect = CountVectorizer(ngram_range=(1, 2), max_df=0.75, min_df=5,
                                 max_features=10000, tokenizer=LemmaTokenizer())
    x_train_counts = count_vect.fit_transform(X_train)
    x_train_tfidf = TfidfTransformer().fit_transform(x_train_counts).toarray()
    v = np.append(labels, x_train_tfidf, axis=1)
    clf = model.fit(v, y_train)
    return clf, count_vect


def predict(model, nlp, features, count_vect, X_test):
    labels = features[SELECTED_FEATURES].to_numpy()
    x_train_counts = count_vect.transform(X_test)
    x_train_tfidf = TfidfTransformer().fit_transform(x_train_counts).toarray()
    v = np.append(labels, x_train_tfidf, axis=1)
    return model.predict(v)


def run_model(path, partitioning_ratios):
    print("Running Logistic Regression model on " + path)

    train = pd.read_csv(path)
    X_data = train['URL'].tolist()
    y_data = train['Verdict'].tolist()
    nlp = spacy.load("en_core_web_sm")

    num = len(X_data)
    last_train_idx = math.floor(partitioning_ratios[0] * num)
    last_validation_idx = last_train_idx + math.floor(partitioning_ratios[1] * num)

    X_train = X_data[:last_train_idx]
    y_train = y_data[:last_train_idx]
    features = pd.read_csv("train_feature.csv")
    compute_features(nlp, features, X_train, "train_feature.csv")
    model = LogisticRegression(max_iter=500, penalty='l1', solver='saga')
    model, count_vect = train_model(model, nlp, features, X_train, y_train)

    X_validation = X_data[last_train_idx+1:last_validation_idx]
    y_validation_answer = y_data[last_train_idx+1:last_validation_idx]
    y_validation_pred = predict(model, nlp, features, count_vect, X_validation)
    validation_score = f1_score(y_validation_answer, y_validation_pred, average='macro')
    print(INDENTATION + 'Score on validation = {}'.format(validation_score))

    X_test = X_data[last_validation_idx + 1:]
    test_features = pd.read_csv('test_feature.csv')
    compute_features(nlp, test_features, X_test, "test_feature.csv")
    y_test_answer = y_data[last_validation_idx + 1:]
    y_test_pred = predict(model, nlp, test_features, count_vect, X_test)
    test_score = f1_score(y_test_answer, y_test_pred, average='macro')
    print(INDENTATION + 'Score on testing = {}'.format(test_score))


run_model("data/1_1_1_1_80000.csv", partitioning_ratios=(0.7, 0.2, 0.1))
run_model("data/1_5_5_1_60000.csv", partitioning_ratios=(0.7, 0.2, 0.1))
run_model("data/1_15_15_1_160000.csv", partitioning_ratios=(0.7, 0.2, 0.1))
