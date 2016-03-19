import numpy as np

def GaussianDistance(vectorA, vectorB, sigma):
	"""
	Computes the distance between two vectors for the Gaussian
	kernel. The two vectors must have the same size.
	"""

	mA = len(vectorA) # length of vectorA
	mB = len(vectorB) # length of vectorB
	
	assert mA == mB, 'The two vectors must have the same size'
	
	A = vectorA.dot(vectorA.T)
	B = vectorB.dot(vectorB.T)
	C = vectorA.dot(vectorB.T)
	
	distance = A - 2*C + B # euclidien distance between two vectors
		
	distance = np.sqrt(2*(1 - np.exp(-distance/(2.*sigma)))) # distance for the Gaussian kernel
	return distance
