import numpy as np

def GaussianDistance(vectorA, vectorB, sigma):
	# Computes the distance between two vectors for the Gaussian
	# kernel. The two vectors must have the same size.

	mA = len(vectorA) # length of vectorA
	mB = len(vectorB) # length of vectorB
	
	assert mA == mB, 'The two vectors must have the same size'
	
	distance = 0
	
	for i in range(mA):
		distance = distance + pow((vectorA[i]-vectorB[i]),2)
		
	distance = np.sqrt(2*(1 - np.exp(-distance/(2.*sigma))))
	return distance
