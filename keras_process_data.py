from __future__ import absolute_import
from __future__ import print_function

import os
import sys
import logging
import re
import nltk
import gensim
import cPickle

import numpy as np
import pandas as pd

from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from collections import defaultdict

# Read data from files
train = pd.read_csv("./data/labeledTrainData.tsv", header=0,
    delimiter="\t", quoting=3)
test = pd.read_csv("./data/testData.tsv", header=0,
    delimiter="\t", quoting=3)
unlabeled_train = pd.read_csv("./data/unlabeledTrainData.tsv", header=0,
    delimiter="\t", quoting=3)

def review_to_wordlist( review, remove_stopwords=False ):
    # Function to convert a document to a sequence of words,
    # optionally removing stop words.  Returns a list of words.
    #
    # 1. Remove HTML
    review_text = BeautifulSoup(review).get_text()
    #  
    # 2. Remove non-letters
    review_text = re.sub("[^a-zA-Z]"," ", review_text)
    #
    # 3. Convert words to lower case and split them
    words = review_text.lower().split()
    #
    # 4. Optionally remove stop words (false by default)
    if remove_stopwords:
        stops = set(stopwords.words("english"))
        words = [w for w in words if not w in stops]
    #
    # 5. Return a list of words
    return(words)

def build_data_train_test(data_train, data_test, train_ratio=0.8):
    """
    Loads data and process data into index
    """
    revs = []
    vocab = defaultdict(float)

    # Pre-process train data set
    for i in xrange(len(data_train)):
        rev = data_train[i]
        y = train['sentiment'][i]
        orig_rev = ' '.join(rev).lower()
        words = set(orig_rev.split())
        for word in words:
            vocab[word] += 1
        datum = {'y': y,
                 'text': orig_rev, 
                 'num_words': len(orig_rev.split()),
                 'split': int(np.random.rand() < train_ratio)}
        revs.append(datum)

    for i in xrange(len(data_test)):
        rev = data_test[i]
        orig_rev = ' '.join(rev).lower()
        words = set(orig_rev.split())
        for word in words:
            vocab[word] += 1
        datum = {'y': -1,
                 'text': orig_rev, 
                 'num_words': len(orig_rev.split()),
                 'split': -1}
        revs.append(datum)

    return revs, vocab

def load_bin_vec(model, vocab, k=300):
    """
    loads 300 x 1 word vecs from Google (Mikolov) pre-trained word2vec
    """
    word_vecs = {}

    for word in vocab.keys():
        try:
            word_vec = model[word]
        except:
            word_vec = np.random.uniform(-0.25, 0.25, k)
        word_vecs[word] = word_vec

    return word_vecs

def get_W(word_vecs, k=300):
    """
    Get word matrix. W[i] is the vector for word indexed by i
    """
    vocab_size = len(word_vecs)
    word_idx_map = dict()
    # position 0 was not used
    W = np.zeros(shape=(vocab_size+1, k), dtype=np.float32)

    i = 1
    for word in word_vecs:
        W[i] = word_vecs[word]
        word_idx_map[word] = i
        i = i + 1
    return W, word_idx_map

if __name__ == '__main__':
    program = os.path.basename(sys.argv[0])
    logger = logging.getLogger(program)
    
    logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
    logging.root.setLevel(level=logging.INFO)
    logger.info("running %s" % ''.join(sys.argv))

    model_file = os.path.join('data', 'GoogleNews-vectors-negative300.bin')
    model = gensim.models.Word2Vec.load_word2vec_format(model_file, binary=True)

    clean_train_reviews = []
    for review in train["review"]:
        clean_train_reviews.append(review_to_wordlist(review, \
            remove_stopwords=True))

    clean_test_reviews = []
    for review in test["review"]:
        clean_test_reviews.append(review_to_wordlist(review, \
            remove_stopwords=True))

    revs, vocab = build_data_train_test(clean_train_reviews, clean_test_reviews)
    max_l = np.max(pd.DataFrame(revs)['num_words'])
    logging.info('data loaded!')
    logging.info('number of sentences: ' + str(len(revs)))
    logging.info('vocab size: ' + str(len(vocab)))
    logging.info('max sentence length: ' + str(max_l))

    w2v = load_bin_vec(model, vocab)
    logging.info('word2vec loaded!')
    logging.info('num words in word2vec: ' + str(len(w2v)))

    W, word_idx_map = get_W(w2v)
    logging.info('extracted index from word2vec! ')

    pickle_file = os.path.join('pickle', 'imdb_train_val_test.pickle')
    cPickle.dump([revs, W, word_idx_map, vocab], open(pickle_file, 'wb'))
    logging.info('dataset created!')