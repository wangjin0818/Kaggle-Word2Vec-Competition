from sklearn.cluster import KMeans
from gensim.models import Word2Vec
from bs4 import BeautifulSoup
from nltk.corpus import stopwords

import time
import numpy as np
import pandas as pd
import re
import nltk

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

clean_train_reviews = []
for review in train["review"]:
	clean_train_reviews.append(review_to_wordlist(review, \
		remove_stopwords=True))

print "Creating average feature for test reviews"
clean_test_reviews = []
for review in test["review"]:
	clean_test_reviews.append(review_to_wordlist(review, \
		remove_stopwords=True))

# load word2vec model
model = Word2Vec.load("300features_40minwords_10context")

start = time.time()

# Set "k" (num_clusters) to be 1/5th of the vocabulary size, or an
# average of 5 words per cluster
word_vectors = model.syn0
num_clusters = word_vectors.shape[0] / 5

# Initialize a k-means object and use it to extract centroids
kmeans_clustering = KMeans(n_clusters=num_clusters)
idx = kmeans_clustering.fit_predict(word_vectors)

# Get the end time and print how long the process took
end = time.time()
elapsed = end - start
print "Time taken for k Means clustring: ", elapsed, "seconds."

# Create a Word / Index dictionary, mapping each vocabulary word to
# a cluster number
word_centroid_map = dict(zip(model.index2word, idx))

for cluster in xrange(0, 10):
	# Print the cluster number
	print "Cluster %d" % cluster

	# Find all of the words for that cluster number, and print them out
	words = []
	for i in xrange(0, len(word_centroid_map.values())):
		if(word_centroid_map.values()[i] == cluster):
			words.append(word_centroid_map.keys()[i])
	print words

def create_bag_of_centroids(wordlist, word_centroid_map):
	# The number of clusters is equal to the highest cluster index
	# in the word / centroid map
	num_centroids = max(word_centroid_map.values()) + 1

	# Pre-allocate the bag of centroids vector (for speed)
	bag_of_centroids = np.zeros(num_centroids, dtype="float32")

	# Loop over the words in the review. If the word is in the vocabulary,
	# find which cluster it belongs to, and increment that cluster count
	# by one
	for word in wordlist:
		if word in word_centroid_map:
			index = word_centroid_map[word]
			bag_of_centroids[index] += 1

	return bag_of_centroids

# Pre-allocate an array for the trainning set bags of centroid (for speed)
train_centroids = np.zeros((train["review"].size, num_clusters), dtype="float32")

# Transform the training set reviews into bags of centroids
counter = 0
for review in clean_train_reviews:
	train_centroids[counter] = create_bag_of_centroids(review, \
	 			word_centroid_map)
	counter += 1

# Repeat for test reviews
test_centroids = np.zeros((test["review"].size, num_clusters), \
	dtype="float32")

counter = 0
for review in clean_test_reviews:
	test_centroids[counter] = create_bag_of_centroids(review, \
	 	word_centroid_map)
	counter += 1

# Fit a random forest to the training data, using 100 trees
from sklearn.ensemble import RandomForestClassifier
forest = RandomForestClassifier(n_estimators=100)

print "Fitting a random forest to labeled training data..."
forest = forest.fit(train_centroids,train["sentiment"])

# Test & extract results
result = forest.predict(test_centroids)

# Write the test results
output = pd.DataFrame(data={"id":test["id"], "sentiment":result})
output.to_csv("./result/BagOfCentroids.csv", index=False, quoting=3)