import numpy as np
from scipy.optimize import fmin_l_bfgs_b
from time import time
from scipy.optimize import minimize

p=28*28
n=1000
x=np.random.randn(p,1)
y=np.random.randn(n,1)
X=np.random.randn(n,p)
alpha=np.random.randn(n,1)

def KK(K_matrix):
	return K_matrix.dot(K_matrix)



def k(x,y):
	return x.T.dot(y)

def K(X):
	#K_matrix=np.asarray([k(X[i],X[j]) for i in range(n) for j in range(n)]).reshape(n,n)
	K_matrix=np.zeros(shape=(n,n))
	for i in range(n):
		for j in range(n):
			K_matrix[i][j]=k(X[i],X[j])
	return K_matrix

def L(alpha,K_matrix,y,lbd=0.5):
	temp=K_matrix.dot(alpha)-y
	temp2=1.0/n*temp.T.dot(temp)+lbd*alpha.T.dot(K_matrix).dot(alpha)
	print temp2[0][0]
	return temp2[0][0]



def Lprime(alpha,K_matrix,y,lbd=0.5):
	#temp=2.0/n*K_matrix.dot(K_matrix)+2*lbd*K_matrix
	temp=2.0/n*KK_matrix+2*lbd*K_matrix
	temp2=temp.dot(alpha)-2.0/n*K_matrix.dot(y)
	return temp2[0]

K_matrix=K(X)
KK_matrix=KK(K_matrix)

fmin_l_bfgs_b(L,alpha,Lprime,args=(K_matrix,y),method='')
