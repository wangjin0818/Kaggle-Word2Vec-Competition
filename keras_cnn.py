import os
import sys
import logging

import cPickle
import numpy as np
import pandas as pd

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten, Reshape, Merge
from keras.layers.embeddings import Embedding
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD, Adadelta
from keras.constraints import unitnorm, maxnorm
from keras.regularizers import l2
from keras.layers.advanced_activations import PReLU

from sklearn.metrics import roc_auc_score

test = pd.read_csv("./data/testData.tsv", header=0,
    delimiter="\t", quoting=3)

def get_idx_from_sent(sent, word_idx_map, max_l=51, kernel_size=5):
    """
    Transforms sentence into a list of indices. Pad with zeroes.
    """
    x = []
    pad = kernel_size - 1
    for i in xrange(pad):
        x.append(0)
    words = sent.split()
    for word in words:
        if word in word_idx_map:
            x.append(word_idx_map[word])

    while len(x) < max_l + 2 * pad:
        x.append(0)

    return x

def make_idx_data(revs, word_idx_map, max_l=51, kernel_size=5):
    """
    Transforms sentences into a 2-d matrix.
    """
    train, test, val = [], [], []
    for rev in revs:
        sent = get_idx_from_sent(rev['text'], word_idx_map, max_l, kernel_size)
        sent.append(rev['y'])
        if rev['split'] == 1:
            train.append(sent)
        elif rev['split'] == 0:
            val.append(sent)
        else:
            test.append(sent)
    train = np.array(train, dtype=np.int)
    val = np.array(val, dtype=np.int)
    test = np.array(test, dtype=np.int)
    return [train, val, test]

if __name__ == '__main__':
    program = os.path.basename(sys.argv[0])
    logger = logging.getLogger(program)
    
    logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
    logging.root.setLevel(level=logging.INFO)
    logger.info(r"running %s" % ''.join(sys.argv))

    logging.info('loading data...')
    pickle_file = os.path.join('pickle', 'imdb_train_val_test.pickle')
    revs, W, word_idx_map, vocab = cPickle.load(open(pickle_file, 'rb'))
    logging.info('data loaded!')

    datasets = make_idx_data(revs, word_idx_map, max_l=1428, kernel_size=5)

    n_train_sample = datasets[0].shape[0]
    logging.info("n_train_sample: %d" % n_train_sample)

    len_sentence = datasets[0].shape[1]     # 1428
    logging.info("len_sentence: %d" % len_sentence)

    num_features = W.shape[1]               # 300
    logging.info("dimension num of word vector: %d" % num_features)

    # train_X = np.zeros((n_train_sample, 1, len_sentence, 300))

    # Train data preparation
    N = datasets[0].shape[0]    # N sentence
    conv_input_width = W.shape[1]      # k = 300
    conv_input_height = int(datasets[0].shape[1] - 1)

    # For each word write a word index (not vector) to X tensor
    train_X = np.zeros((N, conv_input_height), dtype=np.int)
    train_Y = np.zeros((N, 2), dtype=np.int)
    for i in xrange(N):
        for j in xrange(conv_input_height):
            train_X[i, j] = datasets[0][i, j]
        train_Y[i, datasets[0][i, -1]] = 1

    logging.info("train set X shape: " + str(train_X.shape))
    logging.info("train set Y shape: " + str(train_Y.shape))

    # Validation data preparation
    Nv = datasets[1].shape[0]
    # For each word write a word index (not vector) to X tensor
    val_X = np.zeros((Nv, conv_input_height), dtype=np.int)
    val_Y = np.zeros((Nv, 2), dtype=np.int)
    for i in xrange(Nv):
        for j in xrange(conv_input_height):
            val_X[i, j] = datasets[1][i, j]
        val_Y[i, datasets[1][i, -1]] = 1

    logging.info("validation set X shape: " + str(val_X.shape))
    logging.info("validation set Y shape: " + str(val_Y.shape))

    # Test data preparation
    Nt = datasets[2].shape[0]
    # For each word write a word index (not vector) to X tensor
    test_X = np.zeros((Nt, conv_input_height), dtype=np.int)
    for i in xrange(Nv):
        for j in xrange(conv_input_height):
            test_X[i, j] = datasets[2][i, j]

    logging.info("Test set X shape: " + str(test_X.shape))

    # Numver of feature maps (outputs of convolutional layer)
    N_fm = 300

    # kernel size of convolutional layer
    kernel_size = 8

    # print(W.shape)
    # print(conv_input_width, conv_input_height)
    # Keras Model
    model = Sequential()
    # Embedding layer (lookup table of trainable word vectors)
    model.add(Embedding(input_dim=W.shape[0], output_dim=W.shape[1], weights=[W], W_constraint=unitnorm()))
    # Reshape word vectors from Embedding to tensor format suitable for Convolutional layer
    model.add(Reshape(1, conv_input_height, conv_input_width))

    # first convolutional layer
    model.add(Convolution2D(N_fm, 1, kernel_size, conv_input_width, border_mode='valid', W_regularizer=l2(0.0001)))
    # ReLu activation
    model.add(Activation('relu'))

    # aggregate data in every feature map to scalar using MAX operation
    model.add(MaxPooling2D(poolsize=(conv_input_height - kernel_size + 1, 1), ignore_border=True))

    model.add(Flatten())
    model.add(Dropout(0.5))
    # Inner Product layer (as in regular neural network, but without non-linear activation function)
    model.add(Dense(N_fm, 2))
    # SoftMax activation; actually, Dense + SoftMax works as Multinomial Logistic Regression
    model.add(Activation('softmax'))

    # Custom optimizers could be used, though right now standard adadelta is employed
    model.compile(loss='categorical_crossentropy', optimizer='adadelta')

    epoch = 0
    val_acc = []
    val_auc = []
    N_epoch = 10

    for i in xrange(N_epoch):
        model.fit(train_X, train_Y, batch_size=50, nb_epoch=1, verbose=1, show_accuracy=True)
        # output = model.predict_proba(val_X, batch_size=10, verbose=1)
        output = model.predict(val_X, batch_size=50, verbose=1)

        # find validation accuracy using the best threshold value t
        vacc = np.max([np.sum((output[:,1]>t)==(val_Y[:,1]>0.5))*1.0/len(output) for t in np.arange(0.0, 1.0, 0.01)])
        # find validation AUC
        val_Y_result = np.argmax(val_Y, axis=1)
        output_result = np.argmax(output, axis=1)

        print(output_result)
        vauc = roc_auc_score(val_Y_result, output_result)
        val_acc.append(vacc)
        val_auc.append(vauc)
        print 'Epoch {}: validation accuracy = {:.3%}, validation AUC = {:.3%}'.format(epoch, vacc, vauc)
        epoch += 1
        
    print '{} epochs passed'.format(epoch)
    print 'Accuracy on validation dataset:'
    print val_acc
    print 'AUC on validation dataset:'
    print val_auc

    test_Y = model.predict_proba(test_X, batch_size=50, verbose=1)  
    result = np.argmax(test_Y, axis=1)

    result_output = pd.DataFrame(data={"id": test["id"], "sentiment": result})

    # Use pandas to write the comma-separated output file
    result_output.to_csv("./result/cnn.csv", index=False, quoting=3)