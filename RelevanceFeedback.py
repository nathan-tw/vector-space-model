import nltk
import VectorSpace as vs
import numpy as np
from nltk.tokenize import word_tokenize

def nn_and_vb(document):
    feedbackWord = []
    text = word_tokenize(document)
    result = nltk.pos_tag(text)
    for word in result:
        if ('VB' in word[1] or 'NN' in word[1]):
            feedbackWord.append(word)
    return feedbackWord
