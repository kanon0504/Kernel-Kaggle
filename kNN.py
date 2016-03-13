from numpy import *
from GaussianDistance import GaussianDistance
from collections import Counter

def kNN(k, xtr, ytr, xte, sigma):
    # Assigns to the test instance the label of the majority of the labels of the k closest 
	# training examples using the kNN with euclidean distance.
    #
    # Input: k: number of nearest neighbors
    #        xtr: training data           
    #        ytr: class labels of training data
    #        xte: test data
    #        sigma : float. The main parameter of gaussien kernel
    
    
    # Note: To compute the distance between two vectors A and B
    #       use the GaussianDistance(A,B, sigma) function.
    #

    m = asarray(xte).shape[0]
    yte = []

    n = asarray(xtr).shape[0]
    distance = zeros(n)

    for j in range(m):
        for i in range(n):
            distance[i] = GaussianDistance(xtr[i],xte[j], sigma)
    
        index = distance.argsort()[:k]

        neighbors = zeros(k)
        for i in range(k):
            neighbors[i] = ytr[index[i]]

        c = Counter(neighbors)
        yte.append(c.most_common()[0][0])


    # return the label of the test data
    return yte
