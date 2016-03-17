import numpy as np
from load_data import *
from kernel import *
from sklearn.cross_validation import train_test_split
from scipy.optimize import fmin_l_bfgs_b


class svm2:

	def __init__(self, lmb = 0.01, kernel = None, sigma = 1.0, gamma = 0.5):
			
		"""
		:Parameters:
		lmb : float
			The tuning parametre of ridge penalty 
		kernel : string
			Kernel type, can be 'linear' or 'rbf'
		sigma : float
			the main parameter of gaussien kernel, only used when kernel = 'rbf'
		gamma : float
			the main parameter of laplace kernel, only used when kernel = 'laplace'
		"""

		if kernel == None:
			self._kernel = linear_kernel()

		elif kernel == 'linear':
			self._kernel = linear_kernel()

		elif kernel == 'rbf':
			self._kernel = rbf_kernel(sigma = sigma)
			self._sigma = sigma

		elif kernel == 'laplace':
			self._kernel = laplace_kernel(gamma = gamma)
			self._gamma = gamma
			
		if lmb < 0:
		    raise ValueError("lmb must be >= 0")

		self._lmb = float(lmb)        
		self._alpha = None

	def fit(self, X, y):

		"""
		To fit the model:
		input:  X_train, a np.array on shape (n,p)
			y_train, a np.array on shape (n,)
		"""
		self._x = np.asarray(X)
		K_arr = self._kernel.kernel_create(np.asarray(X))
		y_arr = np.asarray(y, dtype=np.float)
		n = K_arr.shape[0]

		def f(alpha,K,lmb,y):
			obj = 2*alpha.dot(y)- alpha.T.dot(K+n*lmb*np.eye(n)).dot(alpha)
			return -obj

		def fprime(alpha,K,lmb,y):
			temp = 2*y-2*(K+n*lmb*np.eye(n)).dot(alpha)
			return -temp
		
		bds = bounds=[(None,None)]*n
		id_y = (y_arr == 1)
		
		for i in range(n):
			if id_y[i]:
				bds[i] = (0,None) 

		alpha0 = np.random.uniform(0,1,(n,1))
		w,_,_ = fmin_l_bfgs_b(f,alpha0,fprime,args=(K_arr,self._lmb,y_arr),bounds = bds)
		self._alpha = w

	def predict(self, Xtest):
		
		"""
		To fit the model:
		input: X_test, a np.array on shape (n,p)
		output: y_train, a np.array on shape (n,)
		"""
		ytemp = self._kernel.Kernel_arr(Xtest)

		return self._alpha.dot(ytemp)

	

if __name__ == '__main__':
	
	xtr0, ytr0 = load_data()
	
	ytr0 = np.asarray([int(i) for i in ytr0])
	xtr0 = flatten(xtr0)
	xtr0 = np.asarray(xtr0).reshape(len(xtr0),28*28)


	
	x_train, x_test, y_train, y_test = train_test_split(xtr0, ytr0, test_size=0.2)
	clf = svm(lmb = 0.4, kernel = 'rbf')
	clf.fit(x_train,y_train)
	y_pre = clf.predict(x_test)
	





