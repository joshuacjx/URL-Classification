from parser import parse
import numpy as np
import pandas as pd
from autocorrect import Speller
import tensorflow as tf
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.models import Sequential
from keras import layers
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
plt.style.use('ggplot')

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

def get_vocab_index_dict(word_freq):
    vocab = set()
    for word in word_freq:
        if word_freq[word] > 1:
            vocab.add(word)
        else:
            vocab.add('UNK')
    vocab_index_dict = {}
    vocab_list = sorted(list(vocab)) # index 1: <UNK>
    for i in range(len(vocab_list)):
        vocab_index_dict[vocab_list[i]] = i + 1 # index 0: <PAD>
    return vocab_index_dict

def word_to_index(w, word_freq, vocab_index_dict):
    if word_freq.get(w, 0) > 1:
        return vocab_index_dict[w]
    else:
        return vocab_index_dict['UNK']

def get_vocab_index_rep(word_rep, word_freq, vocab_index_dict):
    vocab_index_rep = []
    for rep in word_rep:
        vocab_index_rep.append([word_to_index(w, word_freq, vocab_index_dict) for w in rep])
    return vocab_index_rep

df = pd.read_csv('data/1_5_5_1_60000.csv', header=None)
print('Finished reading data frame.')
X_train, X_test, y_train, y_test = train_test_split(df[0].tolist(), np.array(df[1].tolist(), dtype=np.int), train_size=0.7, shuffle=False)
y_train = to_categorical(y_train, 4)
y_test = to_categorical(y_test, 4)

X_train_word_freq, X_train_word_rep = get_word_freq_and_word_rep(X_train)
vocab_index_dict = get_vocab_index_dict(X_train_word_freq)
vocab_size = len(vocab_index_dict) + 1 # including <PAD> and <UNK>
X_train = get_vocab_index_rep(X_train_word_rep, X_train_word_freq, vocab_index_dict)

X_test_word_freq, X_test_word_rep = get_word_freq_and_word_rep(X_test)
X_test = get_vocab_index_rep(X_test_word_rep, X_train_word_freq, vocab_index_dict)

maxlen = 25
X_train = pad_sequences(X_train, padding='post', maxlen=maxlen)
X_test = pad_sequences(X_test, padding='post', maxlen=maxlen)

print('Finished padding.')

# use vocab_index_dict to build word embedding matrix

embedding_dim = 50

def build_embedding_matrix(vocab_index_dict):
    embedding_matrix = np.zeros((vocab_size, embedding_dim), dtype=np.float32)

    with open('glove.6B.50d.txt', 'r') as f:
        for line in f:
            parts = line.split()
            word = parts[0]
            if word in vocab_index_dict:
                index = vocab_index_dict[word]
                vector = np.asarray(parts[1:], dtype=np.float32)
                embedding_matrix[index] = vector

        # <UNK> vector as average of all GloVe vectors
        # retrieved from https://stackoverflow.com/questions/49239941/what-is-unk-in-the-pretrained-glove-vector-files-e-g-glove-6b-50d-txt/53717345#53717345
        avg_glove_vec_str = '-0.12920076 -0.28866628 -0.01224866 -0.05676644 -0.20210965 -0.08389011 0.33359843 0.16045167 0.03867431 0.17833012 0.04696583 -0.00285802 0.29099807 0.04613704 -0.20923874 -0.06613114 -0.06822549 0.07665912 0.3134014 0.17848536 -0.1225775 -0.09916984 -0.07495987 0.06413227 0.14441176 0.60894334 0.17463093 0.05335403 -0.01273871 0.03474107 -0.8123879 -0.04688699 0.20193407 0.2031118 -0.03935686 0.06967544 -0.01553638 -0.03405238 -0.06528071 0.12250231 0.13991883 -0.17446303 -0.08011883 0.0849521 -0.01041659 -0.13705009 0.20127155 0.10069408 0.00653003 0.01685157'
        avg_glove_vec = np.array(avg_glove_vec_str.split())
        unk_index = vocab_index_dict['UNK']
        embedding_matrix[unk_index] = avg_glove_vec

    return embedding_matrix

embedding_matrix = build_embedding_matrix(vocab_index_dict)
print('Finished building embedding matrix with {:.4f} of vocabulary covered.'.format(np.count_nonzero(np.count_nonzero(embedding_matrix, axis=1)) / vocab_size))

# build CNN model

# model.add(layers.Conv2D(filters=32, kernel_size=(50,3), activation='relu', input_shape=(None, maxlen, embedding_dim, 1)))
# model.add(layers.MaxPooling2D(pool_size=(2, 2)))

model = Sequential()
model.add(layers.Embedding(vocab_size, embedding_dim, input_length=maxlen, weights=[embedding_matrix], trainable=False))
model.add(layers.Conv1D(filters=32, kernel_size=3, activation='relu'))
model.add(layers.MaxPooling1D(pool_size=2))
model.add(layers.Dropout(0.2))
model.add(layers.Flatten())
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(4, activation='softmax'))
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=[tf.keras.metrics.AUC()])
model.summary()

# train model

def plot_history(history):
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    x = range(1, len(acc) + 1)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(x, acc, 'b', label='Training acc')
    plt.plot(x, val_acc, 'r', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(x, loss, 'b', label='Training loss')
    plt.plot(x, val_loss, 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()

history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, verbose=1)
loss, score = model.evaluate(X_train, y_train, verbose=False)
print('Training Score: {:.4f}'.format(score))
loss, score = model.evaluate(X_test, y_test, verbose=False)
print('Testing Score:  {:.4f}'.format(score))
plot_history(history)


