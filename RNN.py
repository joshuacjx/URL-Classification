from parser import parse
import numpy as np
import pandas as pd
import tensorflow as tf
from keras import layers
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Masking, Embedding, Bidirectional
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

np.random.seed(42)
tf.random.set_seed(42)

def get_word_freq_and_word_rep(data):
    word_freq = {}
    word_rep = []
    urls = data
    for url in urls:
        parsed = parse(url.lower())
        word_rep.append(parsed)
        for word in parsed:
            word_freq[word] = word_freq.get(word, 0) + 1
    return word_freq, word_rep

def get_vocab_index_dict(word_freq):
    vocab = sorted([w for w in word_freq if word_freq[w] > 1])
    vocab_index_dict = {}
    # vocab = sorted(list(word_freq)) 
    for i in range(len(vocab)):
        vocab_index_dict[vocab[i]] = i + 2 
    vocab_index_dict['UNK'] = 1 # index 0: <PAD>, index 1: <UNK>
    return vocab_index_dict

def word_to_index(word, vocab_index_dict):
    return vocab_index_dict.get(word, vocab_index_dict['UNK'])

def get_vocab_index_rep(word_rep, vocab_index_dict):
    vocab_index_rep = []
    for rep in word_rep:
        vocab_index_rep.append([word_to_index(word, vocab_index_dict) for word in rep])
    return vocab_index_rep

df = pd.read_csv('data/balanced_data_3210.csv', header=None) 
X_train, X_test, y_train, y_test = train_test_split(df[0].tolist(), np.array(df[1].tolist(), dtype=np.int), test_size=0.2, random_state=42)
y_train = to_categorical(y_train, 4)
y_test = to_categorical(y_test, 4)

X_train_word_freq, X_train_word_rep = get_word_freq_and_word_rep(X_train)
vocab_index_dict = get_vocab_index_dict(X_train_word_freq)
vocab_size = len(vocab_index_dict) + 1 # including <PAD> and <UNK>
X_train = get_vocab_index_rep(X_train_word_rep, vocab_index_dict)

X_test_word_freq, X_test_word_rep = get_word_freq_and_word_rep(X_test)
X_test = get_vocab_index_rep(X_test_word_rep, vocab_index_dict)

maxlen = 20
X_train = pad_sequences(X_train, padding='post', maxlen=maxlen)
X_test = pad_sequences(X_test, padding='post', maxlen=maxlen)

print('Finished preprocessing.')

embedding_dim = 50

def build_embedding_matrix(vocab_index_dict):
    embedding_matrix = np.zeros((vocab_size, embedding_dim), dtype=np.float32)

    with open('data/glove.6B.50d.txt', 'r') as f:
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

# build RNN model
def build_model(dropout_rate, recurrent_dropout, n_dense_1, n_dense_2, n_dense_3):
    model = Sequential()
    model.add(Embedding(vocab_size, embedding_dim, input_length=maxlen, weights=[embedding_matrix], trainable=False))
    model.add(Masking(mask_value=0.0))
    
    # Recurrent layer
    # model.add(LSTM(128, return_sequences=True))
    model.add(LSTM(256, return_sequences=False, dropout=dropout_rate, recurrent_dropout=recurrent_dropout))
    # with bidirectional gate
    # model.add(Bidirectional(LSTM(256, return_sequences=False, dropout=dropout_rate, recurrent_dropout=recurrent_dropout)))
    
    # model.add(Dropout(0.1))
    model.add(Dense(n_dense_1, activation='relu'))
    model.add(Dense(n_dense_2, activation='relu'))
    model.add(Dense(n_dense_3, activation='relu'))
    model.add(layers.Dense(4, activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=[tf.keras.metrics.AUC()])
        # metrics=['accuracy'])
    return model

# train model, change dropout rates and nodes in dense layer
model = build_model(0.2, 0.2, 128, 64, 16)

# Create callbacks
callbacks = [EarlyStopping(monitor='val_loss', patience=5)]
# ModelCheckpoint('../models/model.h5', save_best_only=True, save_weights_only=False)]
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=50, batch_size=32, verbose=1, callbacks=callbacks)

# get scores for validation and test, labels should be in one-hot vector
y_val = model.predict(X_train)
y_pred = model.predict(X_test)
score1 = roc_auc_score(y_train, y_val, average = None, multi_class='ovo')
score2 = roc_auc_score(y_test, y_pred, average = None, multi_class='ovo')
print('Training Score: ' + str(score1))
print('Test Score: ' + str(score2))

'''
epoch = 10 
R: 128 64, D: 32, 8
without gate
Training Score: [0.97260869 0.9097496  0.93526084 0.95251548]
Test Score: [0.94137129 0.82715933 0.86920181 0.88803467]

with gate
Training Score: [0.9752866  0.91479725 0.93969579 0.95414356]
Test Score: [0.93902991 0.82900618 0.87039235 0.88805683]

R:64 D: 32, 16, 8 with gate epoch = 20
Training Score: [0.95900786 0.87233514 0.90697067 0.92530187]
Test Score: [0.94181862 0.83278358 0.87534123 0.89222665]

without drop out 0.1: epoch stops at 22 R:64 D: 32, 16, 8 
Training Score: [0.96255468 0.88018314 0.91255899 0.93061194]
Test Score: [0.94213832 0.8325654  0.87376838 0.88931592]

without drop out 0.1: epoch stops at 15 R:128 D: 64, 32, 16
Training Score: [0.97005699 0.90348209 0.93050197 0.94687537]
Test Score: [0.94282869 0.83387974 0.87682298 0.89454678]

epoch stops at 11 R:256, D: 128, 64, 16
Training Score: [0.97900612 0.92508984 0.9477194  0.96178847]
Test Score: [0.94435102 0.83514778 0.87758712 0.89326018]

without gate epoch stops at 14, rest same as above
Training Score: [0.97490824 0.91381751 0.94052618 0.95386801]
Test Score: [0.94155902 0.83417657 0.87512293 0.89391336]

with gate maxlex = 20
Training Score: [0.9815753  0.93365642 0.95372632 0.966477]
Test Score: [0.94438574 0.83706692 0.88068646 0.89503688]

batch_size = 32 -> used
aining Score: [0.98186493 0.93505457 0.9549567  0.96672216]
Test Score: [0.94435581 0.83803908 0.88019144 0.89690211]

maxlen = 18
Training Score: [0.98272759 0.93542779 0.95690418 0.96877337]
Test Score: [0.94402607 0.83727384 0.88107101 0.89632157]

maxlen = 19
Training Score: [0.98244954 0.93488131 0.95620345 0.96773935]
Test Score: [0.94387098 0.8374998  0.87745103 0.89409063]

without gate -> used
Training Score: [0.97453554 0.91128087 0.93903356 0.95416133]
Test Score: [0.94499588 0.83557261 0.8783797  0.89541279]
'''
