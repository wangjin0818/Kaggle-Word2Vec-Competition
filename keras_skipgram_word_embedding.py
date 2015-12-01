#from __future__ import absolute_import
#from __future__ import print_function

import numpy as np
import theano
import os
import re
import json

from six.moves import cPickle
# keras import 
from keras.preprocessing import sequence, text
from keras.optimizers import SGD, RMSprop, Adagrad
from keras.utils import np_utils, generic_utils
from keras.models import Sequential
from keras.layers.embeddings import WordContextProduct, Embedding

max_features = 50000 # vocabulary size: top 50,000 most common words in data
skip_top = 100 # ignore top 100 most common words
nb_epoch = 1
dim_proj = 256 # embedding space dimension

save = True
load_model = False
load_tokenizer = False
train_model = True

save_dir = os.path.join('.', 'models')
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
model_load_fname = "HN_skipgram_model.pkl"
model_save_fname = "HN_skipgram_model.pkl"
tokenizer_fname = "HN_tokenizer.pkl"

data_path = os.path.join('.', 'data', 'HNCommentsAll.1perline.json')

# text preprocessing utils
html_tags = re.compile(r'<.*?>')
to_replace = [('&#x27;', "'")]
hex_tags = re.compile(r'&.*?;')

def clean_comment(comment):
    c = str(comment.encode("utf-8"))
    c = html_tags.sub(' ', c)
    for tag, char in to_replace:
        c = c.replace(tag, char)
    c = hex_tags.sub(' ', c)
    return c

def text_generator(path = data_path):
    f = open(path)
    for i, l in enumerate(f):
        comment_data = json.loads(l)
        comment_text = comment_data["comment_text"]
        comment_text = clean_comment(comment_text)
        if i % 10000 == 0:
            print i
        yield comment_text
    f.close()

if load_tokenizer:
    print 'Load tokenizer...'
    tokenizer = cPickle.load(open(os.path.join(save_dir, tokenizer_fname), 'rb'))
else:
    print "Fit tokenizer..."
    tokenizer = text.Tokenizer(nb_words=max_features)
    tokenizer.fit_on_texts(text_generator())
    if save:
        print("Save tokenizer...")
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        cPickle.dump(tokenizer, open(os.path.join(save_dir, tokenizer_fname), "wb"))

if train_model:
    if load_model:
        print 'load model...'
        model = cPickle.load(open(os.path.join(save_dir, model_load_fname), 'rb'))
    else:
        print 'build model...'
        model = Sequential()
        model.add(WordContextProduct(max_features, proj_dim=dim_proj, init="uniform"))
        model.compile(loss='mse', optimizer='rmsprop')

    sampling_table = sequence.make_sampling_table(max_features)

    for e in range(nb_epoch):
        print '-' * 40
        print 'Epoch' + str(e)
        print '-' * 40

        progbar = generic_utils.Progbar(tokenizer.document_count)
        samples_seen = 0
        losses = []

        for i, seq in enumerate(tokenizer.texts_to_sequences_generator(text_generator())):
            # get skipgram couples for one text in the dataset
            couples, labels = sequence.skipgrams(seq, max_features, window_size=4, negative_samples=1., sampling_table=sampling_table)
            if couples:
                # one gradient update per sentence (one sentence = a few 1000s of word couples)
                X = np.array(couples, dtype="int32")
                loss = model.train(X, labels)
                losses.append(loss)
                if len(losses) % 100 == 0:
                    progbar.update(i, values=[("loss", np.mean(losses))])
                    losses = []
                    samples_seen += len(labels)
        print 'Samples seen:' + str(samples_seen)
    print 'Training completed!' 