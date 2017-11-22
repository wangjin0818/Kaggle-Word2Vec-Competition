import os
import sys
import logging

import pickle
import numpy as np
import pandas as pd

from keras.models import Sequential, Model
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.embeddings import Embedding
from keras.layers.convolutional import Convolution1D, MaxPooling1D
from keras.optimizers import SGD, Adadelta
# from keras.layers.recurrent import LSTM, GRU
from keras.regularizers import l2
from keras.layers.advanced_activations import PReLU
from keras.preprocessing import sequence
from keras.layers import Input, merge
from keras.optimizers import RMSprop, Adagrad, Adadelta

from keras.utils import np_utils


# from sklearn.metrics import roc_auc_score

batch_size = 50
nb_epoch = 10
hidden_dim = 120

kernel_size = 3
nb_filter = 120

test = pd.read_csv("./data/testData.tsv", header=0,
    delimiter="\t", quoting=3)

def get_idx_from_sent(sent, word_idx_map):
    """
    Transforms sentence into a list of indices. Pad with zeroes.
    """
    x = []
    words = sent.split()
    for word in words:
        if word in word_idx_map:
            x.append(word_idx_map[word])
        else:
            x.append(1)

    return x

def make_idx_data(revs, word_idx_map, maxlen):
    """
    Transforms sentences into a 2-d matrix.
    """
    # train, test, val = [], [], []
    X_train, X_test, X_dev, y_train, y_dev = [], [], [], [], []
    for rev in revs:
        sent = get_idx_from_sent(rev['text'], word_idx_map)
        y = rev["y"]

        sent.append(rev['y'])
        # train set
        if rev['split'] == 1:
            X_train.append(sent)
            y_train.append(y)

        # dev set
        elif rev['split'] == 0:
            X_dev.append(sent)
            y_dev.append(y)

        # test set
        elif rev['split'] == -1:
            X_test.append(sent)

    X_train = sequence.pad_sequences(np.array(X_train), maxlen=maxlen)
    X_dev = sequence.pad_sequences(np.array(X_dev), maxlen=maxlen)
    X_test = sequence.pad_sequences(np.array(X_test), maxlen=maxlen)
    y_train = np_utils.to_categorical(np.array(y_train))
    y_dev = np_utils.to_categorical(np.array(y_dev))
    
    return [X_train, X_test, X_dev, y_train, y_dev]

if __name__ == '__main__':
    program = os.path.basename(sys.argv[0])
    logger = logging.getLogger(program)
    
    logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
    logging.root.setLevel(level=logging.INFO)
    logger.info(r"running %s" % ''.join(sys.argv))

    logging.info('loading data...')
    pickle_file = os.path.join('pickle', 'imdb_train_val_test.pickle3')
    revs, W, word_idx_map, vocab, max_l = pickle.load(open(pickle_file, 'rb'))
    logging.info('data loaded!')

    maxlen = 120
    X_train, X_test, X_dev, y_train, y_dev = make_idx_data(revs, word_idx_map, maxlen=maxlen)

    n_train_sample = X_train.shape[0]
    logging.info("n_train_sample [n_train_sample]: %d" % n_train_sample)

    n_test_sample = X_test.shape[0]
    logging.info("n_test_sample [n_train_sample]: %d" % n_test_sample)

    len_sentence = X_train.shape[1]     # 200
    logging.info("len_sentence [len_sentence]: %d" % len_sentence)

    max_features = W.shape[0]
    logging.info("max features of word vector [max_features]: %d" % max_features)

    num_features = W.shape[1]               # 400
    logging.info("dimension num of word vector [num_features]: %d" % num_features)

    # Keras Model
    # this is the placeholder tensor for the input sequence
    sequence = Input(shape=(maxlen, ), dtype='int32')

    embedded = Embedding(input_dim=max_features, output_dim=num_features, input_length=maxlen, weights=[W], trainable=False) (sequence)
    embedded = Dropout(0.25) (embedded)

    # convolutional layer
    convolution = Convolution1D(filters=nb_filter,
                            kernel_size=kernel_size,
                            padding='valid',
                            activation='relu',
                            strides=1
                            ) (embedded)

    maxpooling = MaxPooling1D(pool_size=2) (convolution)
    maxpooling = Flatten() (maxpooling)

    # We add a vanilla hidden layer:
    dense = Dense(120) (maxpooling)    # best: 120
    dense = Dropout(0.25) (dense)    # best: 0.25
    dense = Activation('relu') (dense)

    output = Dense(2, activation='softmax') (dense)
    model = Model(inputs=sequence, outputs=output)

    # optimizer = RMSprop()
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

    model.fit(X_train, y_train, validation_data=[X_dev, y_dev], batch_size=batch_size, epochs=nb_epoch, verbose=2)

    y_test = model.predict(X_test, batch_size=batch_size, verbose=2)  
    result = np.argmax(y_test, axis=1)

    result_output = pd.DataFrame(data={"id": test["id"], "sentiment": result})
    # Use pandas to write the comma-separated output file
    result_output.to_csv("./result/cnn.csv", index=False, quoting=3)