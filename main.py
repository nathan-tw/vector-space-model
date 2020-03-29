import VectorSpace
import ReadFiles
import numpy as np
import util
import RelevanceFeedback as rf


def sortByRatings(lst):
    return lst[1]

#print result in a format
def printResult(top5Lst):
    print('{}     {}'.format('DocID', 'Score'))
    for each in top5Lst:
        print('{}    {}'.format(each[0], each[1]))
    print('')

if __name__ == '__main__':

    # read files from 'documents. directory
    indexes, contents = ReadFiles.readDocuments()
    

    # new a vector space model
    vectorspace = VectorSpace.VectorSpace(contents)
    

    # search a test query
    query = 'drill wood sharp'
    queryVector = np.array(vectorspace.makeTfidfVector(query))
    [tf_cos, tf_dist] = vectorspace.searchTf(query)
    [tfidf_cos, tfidf_dist] = vectorspace.searchTfidf(query)

    # bind indexes to ratings then sort
    top5_tf_cos = sorted(list(zip(indexes, tf_cos)), reverse=True, key=sortByRatings)[:5]
    top5_tf_dist = sorted(list(zip(indexes, tf_dist)), reverse=False, key=sortByRatings)[:5]
    top5_tfidf_cos = sorted(list(zip(indexes, tfidf_cos)), reverse=True, key=sortByRatings)[:5]
    top5_tfidf_dist = sorted(list(zip(indexes, tfidf_dist)), reverse=False, key=sortByRatings)[:5]
    
    print('Term Frequency Weighting + Cosine Similarity:')
    printResult(top5_tf_cos)

    print('Term Frequency Weighting + Euclidean Distance:')
    printResult(top5_tf_dist)

    print('TF-IDF Weighting + Cosine Similarity:')
    printResult(top5_tfidf_cos)

    print('TF-IDF Weighting + Euclidean Distance:')
    printResult(top5_tfidf_dist)

    feedBackIndex = [indexes.index(feedback[0]) for feedback in top5_tfidf_cos]
    feedBackDocument = [contents[index] for index in feedBackIndex]
    scores = []
    for doc in feedBackDocument:
        feedbackWord = rf.nn_and_vb(doc)
        feedbackVector = np.array(vectorspace.makeTfidfVector(' '.join(feedbackWord)))*0.5
        qfVector = queryVector+feedbackVector
        finalScore = util.cosine(qfVector, queryVector)
        scores.append(finalScore)
    relevanceFeedback = sorted(list(zip(feedBackIndex, scores)), reverse=True, key=sortByRatings)

    print('Feedback Queries + TF-IDF Weighting + Cosine Similarity:')
    printResult(relevanceFeedback)
    
