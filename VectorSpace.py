from pprint import pprint
from Parser import Parser
from textblob import TextBlob as tb
import numpy as np
import util
import nltk

class VectorSpace:
    """ A algebraic model for representing text documents as vectors of identifiers. 
    A document is represented as a vector. Each dimension of the vector corresponds to a 
    separate term. If a term occurs in the document, then the value in the vector is non-zero.
    """
    documents =[]
    #Collection of document term vectors
    tfVectors = []
    tfidfVectors = []

    #Mapping of vector index to keyword
    vectorKeywordIndex=[]

    #Tidies terms
    parser=None


    def __init__(self, documents=[]):
        self.documents = documents
        self.tfVectors=[]
        self.tfidfVectors=[]
        self.parser = Parser()
        if(len(documents)>0):
            self.build(documents)

    def build(self,documents):
        """ Create the vector space for the passed document strings """
        self.vectorKeywordIndex = self.getVectorKeywordIndex(documents)
        self.tfVectors = [self.makeTfVector(document) for document in documents]
        self.tfidfVectors = [self.makeTfidfVector(document) for document in documents]
        



    def getVectorKeywordIndex(self, documentList):
        """ create the keyword associated to the position of the elements within the document vectors """

        #Mapped documents into a single word string	
        vocabularyString = " ".join(documentList)

        vocabularyList = self.parser.tokenise(vocabularyString)
        #Remove common words which have no search value
        vocabularyList = self.parser.removeStopWords(vocabularyList)
        uniqueVocabularyList = util.removeDuplicates(vocabularyList)

        vectorIndex={}
        offset=0
        #Associate a position with the keywords which maps to the dimension on the vector used to represent this word
        for word in uniqueVocabularyList:
            vectorIndex[word]=offset
            offset+=1
        return vectorIndex  #(keyword:position)
    
    # function that make a vector by term frequency weighting
    def makeTfVector(self, wordString):
        """ @pre: unique(vectorIndex) """

        #Initialise vector with 0's
        vector = [0] * len(self.vectorKeywordIndex)
        wordList = self.parser.tokenise(wordString)
        wordList = self.parser.removeStopWords(wordList)
        wordSet = set(wordList)
        for word in wordSet:
            tf = util.tf(word, wordList)
            vector[self.vectorKeywordIndex[word]]+=tf
        return vector


    # function that make a vector by tf-idf weighting
    def makeTfidfVector(self, wordString):
        """ @pre: unique(vectorIndex) """

        #Initialise vector with 0's
        vector = [0] * len(self.vectorKeywordIndex)
        wordList = self.parser.tokenise(wordString)
        wordList = self.parser.removeStopWords(wordList)
        wordSet = set(wordList)
        for word in wordSet:
            try:
                tfidf = util.tfidf(word, wordList, self.documents)
                vector[self.vectorKeywordIndex[word]]+=tfidf
            except: 
                continue
        return vector

    #get similarity and distance from term frequency vector
    def searchTf(self, query):
        """ search for documents that match based on a list of terms """
        queryVector = self.makeTfVector(query)
        tf_cos = [util.cosine(queryVector, documentVector) for documentVector in self.tfVectors]
        tf_dist = [util.euclidean(queryVector, documentVector) for documentVector in self.tfVectors]
        return [tf_cos, tf_dist]

    #get similarity and distance from tf-idf vector
    def searchTfidf(self, query):
        """ search for documents that match based on a list of terms """
        queryVector = self.makeTfidfVector(query)
        tfidf_cos = [util.cosine(queryVector, documentVector) for documentVector in self.tfidfVectors]
        tfidf_dist = [util.euclidean(queryVector, documentVector) for documentVector in self.tfidfVectors]
        return [tfidf_cos, tfidf_dist]

    def getRelevanceFeedbackVector(self, wordString):
        """ @pre: unique(vectorIndex) """
        #Initialise vector with 0's
        wordList = self.parser.tokenise(wordString)
        wordList = self.parser.removeStopWords(wordList)
        result = nltk.pos_tag(wordList)
        feedbackWord = []
        for word in result:
            if ('VB' in word[1] or 'NN' in word[1]):
                feedbackWord.append(word[0])
        return np.array(self.makeTfidfVector(' '.join(feedbackWord)))*0.5
