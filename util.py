import sys
import math
import numpy as np
from textblob import TextBlob as tb


#http://www.scipy.org/
try:
	from numpy import dot
	from numpy.linalg import norm
except:
	print("Error: Requires numpy from http://www.scipy.org/. Have you installed scipy?")
	sys.exit() 

def removeDuplicates(list):
	""" remove duplicates from a list """
	return set((item for item in list))

def tf(word, wordlist):
    return wordlist.count(word)

def n_containing(word, documents):
	return sum(1 for doc in documents if word in doc)

def idf(word, documents):
	return math.log(len(documents) / (1 + n_containing(word, documents)))

def tfidf(word, wordlist, documents):
    return tf(word, wordlist) * idf(word, documents)

def cosine(vector1, vector2):
	return float(dot(vector1,vector2) / (norm(vector1) * norm(vector2)))

def euclidean(vector1, vector2):
	vec1 = np.array(vector1)
	vec2 = np.array(vector2)
	dist = norm(vec1-vec2)
	return float(dist)
