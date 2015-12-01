input_list = ['all', 'this', 'happened', 'more', 'or', 'less']

def find_bigrams_old(input_list):
	bigram_list = []
	for i in range(len(input_list) - 1):
		bigram_list.append((input_list[i], input_list[i+1]))
	return bigram_list

def find_bigrams(input_list):
	return zip(input_list, input_list[1:])

def find_trigrams(input_list):
	return zip(input_list, input_list[1:], input_list[2:])

def find_unigrams(intput_list):
	return zip(input_list)

print find_unigrams(input_list)
print find_bigrams(input_list)
print find_trigrams(input_list)

def find_ngrams(input_list, n):
	return zip(*[input_list[i:] for i in range(n)])

print find_ngrams(input_list, 1)