import os
import gensim
import re
import multiprocessing
import logging
import sys

import pandas as pd
import numpy as np

from bs4 import BeautifulSoup
from gensim.models import Doc2Vec
from gensim.models.doc2vec import LabeledSentence
from nltk.corpus import stopwords
# import gensim.models.Doc2Vec.LabeledSentence as LabeledSentence
from sklearn.ensemble import RandomForestClassifier

train = pd.read_csv("./data/labeledTrainData.tsv", header=0,
    delimiter="\t", quoting=3)
test = pd.read_csv("./data/testData.tsv", header=0,
    delimiter="\t", quoting=3)
unlabeled_train = pd.read_csv("./data/unlabeledTrainData.tsv", header=0,
    delimiter="\t", quoting=3)

num_features = 300

## print model["good"]
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

def getParagraphVectors(model, num, prefix="TRAIN_"):
    data = np.zeros((num, num_features))
    print num

    for i in range(num):
        paragraph_string = prefix + str(i)
        print paragraph_string
        paragraph_vector = model.docvecs[paragraph_string]
        for j in range(num_features):
            data[i, j] = paragraph_vector[j]
    return data

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
    sentences = LabeledLineSentence(total_reviews)

    model = Doc2Vec(size=num_features, workers=multiprocessing.cpu_count())
    model.build_vocab(sentences.to_array())
    for epoch in range(10):
        model.train(sentences)

    trainData = getParagraphVectors(model, num=num_train, prefix="TRAIN_")
    testData = getParagraphVectors(model, num=num_test, prefix="TEST_")
    
    # train a classifier
    rfClf = RandomForestClassifier(n_estimators=500)
    rfClf.fit(trainData, train["sentiment"])

    result = rfClf.predict(testData)
    output = pd.DataFrame(data={"id":test["id"], "sentiment":result})
    output.to_csv("./result/Paragraph_vector.csv", index=False, quoting=3)