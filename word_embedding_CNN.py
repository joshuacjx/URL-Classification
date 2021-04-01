from parser import parse
import pandas as pd
from autocorrect import Speller
import numpy as np
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split

def get_word_freq_and_word_rep(data):
    word_freq = {}
    word_rep = []
    sp = Speller(lang='en')
    urls = data
    for url in urls:
        parsed_corrected = [sp(word) for word in parse(url.lower())]
        word_rep.append(parsed_corrected)
        for word in parsed_corrected:
            word_freq[word] = word_freq.get(word, 0) + 1
    return word_freq, word_rep

def get_vocab(word_freq):
    vocab = set()
    for word in word_freq:
        if word_freq[word] > 1:
            vocab.add(word)
        else:
            vocab.add('UNK')
    return vocab

def get_vocab_index_dict(vocab):
# vocab = get_vocab(word_freq)
    vocab_index = {}
    vocab_list = sorted(list(vocab))
    for i in range(len(vocab_list)):
        vocab_index[vocab_list[i]] = i + 1 # index 0: <PAD>
    return vocab_index

def word_to_index(w, vocab_index_dict):
    if w in vocab_index_dict:
        return vocab_index_dict[w]
    else:
        return vocab_index_dict['UNK']

def get_vocab_index_rep(word_rep, vocab_index_dict):
    vocab_index_rep = []
    for rep in word_rep:
        vocab_index_rep.append([word_to_index(w, vocab_index_dict) for w in rep])
    return vocab_index_rep

df = pd.read_csv('data/1_5_5_1_60000.csv', header=None).head(20)
word_freq, word_rep = get_word_freq_and_word_rep(df[0].tolist())
vocab = get_vocab(word_freq)
vocab_index = get_vocab_index_dict(vocab)
vocab_index_rep = get_vocab_index_rep(word_rep, vocab_index)

X_train, X_test, y_train, y_test = train_test_split(vocab_index_rep, df[1].tolist(), train_size=0.7, shuffle=False)

maxlen = 15
X_train = pad_sequences(X_train, padding='post', maxlen=maxlen)
X_test = pad_sequences(X_test, padding='post', maxlen=maxlen)
print(X_train)

# use word index to make word embedding vector matrix


# embeddings_dict = {}
# avg_glove_vec_str = '-0.12920076 -0.28866628 -0.01224866 -0.05676644 -0.20210965 -0.08389011 0.33359843 0.16045167 0.03867431 0.17833012 0.04696583 -0.00285802 0.29099807 0.04613704 -0.20923874 -0.06613114 -0.06822549 0.07665912 0.3134014 0.17848536 -0.1225775 -0.09916984 -0.07495987 0.06413227 0.14441176 0.60894334 0.17463093 0.05335403 -0.01273871 0.03474107 -0.8123879 -0.04688699 0.20193407 0.2031118 -0.03935686 0.06967544 -0.01553638 -0.03405238 -0.06528071 0.12250231 0.13991883 -0.17446303 -0.08011883 0.0849521 -0.01041659 -0.13705009 0.20127155 0.10069408 0.00653003 0.01685157'
# avg_glove_vec = np.array(avg_glove_vec_str.split(' '))

# with open("data/glove.6B.50d.txt", 'r') as f:
#     for line in f:
#         values = line.split()
#         word = values[0]
#         if word in vocab:
#             vector = np.asarray(values[1:], dtype=np.float32)
#             embeddings_dict[word] = vector
#     embeddings_dict['UNK'] = avg_glove_vec
#     embeddings_dict['PAD'] = np.zeros(50, dtype=np.float32)


# from sklearn.model_selection import train_test_split

# sentences = df_yelp['sentence'].values
# y = df_yelp['label'].values
# sentences_train, sentences_test, y_train, y_test = train_test_split(sentences, y, test_size=0.25, random_state=1000)

# from sklearn.feature_extraction.text import CountVectorizer

# vectorizer = CountVectorizer()
# vectorizer.fit(sentences_train)

# X_train = vectorizer.transform(sentences_train)
# X_test  = vectorizer.transform(sentences_test)

# from keras.models import Sequential
# from keras import layers

# input_dim = X_train.shape[1]  # Number of features

# model = Sequential()
# model.add(layers.Dense(10, input_dim=input_dim, activation='relu'))
# model.add(layers.Dense(1, activation='sigmoid'))
# model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# model.summary()




# embedding_dim = 100

# def create_model(num_filters, kernel_size, vocab_size, embedding_dim, maxlen):
#     model = Sequential()
#     model.add(layers.Embedding(vocab_size, embedding_dim, input_length=maxlen))
#     model.add(layers.Conv1D(num_filters, kernel_size, activation='relu'))
#     model.add(layers.GlobalMaxPooling1D())
#     model.add(layers.Dense(10, activation='relu'))
#     model.add(layers.Dense(1, activation='sigmoid'))
#     model.compile(optimizer='adam',
#                   loss='binary_crossentropy',
#                   metrics=['accuracy'])
#     return model

# model = Sequential()
# model.add(layers.Embedding(vocab_size, embedding_dim, input_length=maxlen))
# model.add(layers.Conv1D(128, 5, activation='relu'))
# model.add(layers.GlobalMaxPooling1D())
# model.add(layers.Dense(10, activation='relu'))
# model.add(layers.Dense(1, activation='sigmoid'))
# model.compile(optimizer='adam',
#               loss='binary_crossentropy',
#               metrics=['accuracy'])
# model.summary()

# history = model.fit(X_train, y_train,
#                     epochs=10,
#                     verbose=False,
#                     validation_data=(X_test, y_test),
#                     batch_size=10)
# loss, accuracy = model.evaluate(X_train, y_train, verbose=False)
# print("Training Accuracy: {:.4f}".format(accuracy))
# loss, accuracy = model.evaluate(X_test, y_test, verbose=False)
# print("Testing Accuracy:  {:.4f}".format(accuracy))




