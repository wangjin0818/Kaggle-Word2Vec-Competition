import os
import sys
import logging
import re
import codecs

import numpy as np
import pandas as pd

from bs4 import BeautifulSoup
from gensim.models import Doc2Vec
from gensim.models.doc2vec import LabeledSentence
from nltk.corpus import stopwords

train = pd.read_csv("./data/labeledTrainData.tsv", header=0,
    delimiter="\t", quoting=3)
test = pd.read_csv("./data/testData.tsv", header=0,
    delimiter="\t", quoting=3)
unlabeled_train = pd.read_csv("./data/unlabeledTrainData.tsv", header=0,
    delimiter="\t", quoting=3)

num_features = 300

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

class LabeledLineSentence(object):
    def __init__(self, sources):
        self.sources = sources

    def __iter__(self):
        for tag, line in self.sources:
            yield LabeledSentence(words=line, tags=[tag])

    def to_array(self):
        self.sentences = []
        for tag, line in self.sources:
            self.sentences.append(LabeledSentence(words=line, tags=[tag]))

        return self.sentences

    def sentences_perm(self):
        return np.random.permutation(self.sentences)
    

if __name__ == '__main__':
    program = os.path.basename(sys.argv[0])
    logger = logging.getLogger(program)
    
    logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
    logging.root.setLevel(level=logging.INFO)
    logger.info("running %s" % ''.join(sys.argv))

    clean_train_reviews = []
    train_id = 0
    for review in train["review"]:
        tag = "TRAIN_" + str(train_id)
        clean_train_reviews.append([tag, review_to_wordlist(review, \
            remove_stopwords=True)])
        train_id = train_id + 1

        if train_id % 1000 == 0:
            logging.info("load train data: %d" % (train_id))

    clean_test_reviews = []
    test_id = 0
    for review in test["review"]:
        tag = "TEST_" + str(test_id)
        clean_test_reviews.append([tag, review_to_wordlist(review, \
            remove_stopwords=True)])
        test_id = test_id + 1

        if test_id % 1000 == 0:
            logging.info("load test data: %d" % (test_id))

    num_train = len(clean_train_reviews)
    logging.info("train data length: %d" % (num_train))
    num_test = len(clean_test_reviews)
    logging.info("test data length: %d" % (num_test))

    total_reviews = clean_train_reviews + clean_test_reviews

    # load enwiki
    with codecs.open(os.path.join('data', 'wiki.en.text'), 'r', 'utf-8') as fr:
        i = 1
        for line in fr.readlines():
            line = line.strip().split(" ")
            line_name = "wiki_" + str(i)
            total_reviews.append([line_name, line])
            i = i + 1

            if(i % 10000 == 0):
                logger.info("Load " + str(i) + " articles")

    print('hello')