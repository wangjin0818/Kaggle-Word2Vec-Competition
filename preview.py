# Import the pandas package, then use the "read_csv" function to read
# the labeled training data
import pandas as pd
train = pd.read_csv("./data/labeledTrainData.tsv", header=0, \
					delimiter="\t", quoting=3)

## print train.shape
## print train.columns.values
## print train["review"][0]

# Import BeautifulSoup into workspace
from bs4 import BeautifulSoup

# Initialize the BeautifulSoup object on a single movie review
example1 = BeautifulSoup(train["review"][0])

# Print the raw review and then the output of get_text(). for
# comparison
## print train["review"][0]
## print example1.get_text()

import re
# Use regular repressions to do a find-and-replace
letters_only = re.sub("[^a-zA-Z]", " ", example1.get_text())
## print letters_only

lower_case = letters_only.lower()
words = lower_case.split()
## print words

# Import the stop word list
## import nltk
## nltk.download()

from nltk.corpus import stopwords
## print stopwords.words("english")
words = [w for w in words if not w in stopwords.words("english")]
print words