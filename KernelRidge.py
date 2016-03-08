import numpy as np
from load_data import load_data, plot, flatten
from sklearn.cross_validation import train_test_split
import time


class linear_kernel:

    def kernel_create(self, x):
    	"""
    	create the kernel matrix
    	"""
        self._x = x
        self._Kmatrix = x.dot(x.T)
        return self._Kmatrix
    
    def Kernel_arr(self,Xtest):
		"""
		For prediction: 
		input x_test, np.array on shape (m, p)
		output np.array, on shape (n,m)
		"""
		return self._x.dot(Xtest.T)



class rbf_kernel:

	def __init__(self, sigma = 1.0):
		
		"""
		:parameters:
		sigma : float
			The main parameter of gaussien kernel
		"""
		self._sigma = sigma


	def kernel_create(self, x):
		
		"""
		create the kernel matrix
		input : X_train, a np.array on shape (n,p)
		"""
		x = np.asarray(x)
		n = x.shape[0]
		self._x = x

		A = 2*x.dot(x.T)
		B = np.linalg.norm(x,axis = 1)[:,np.newaxis].dot(np.ones(shape=(1,n)))
		B *= B
		C = np.exp(A-B-B.T)/(2*self._sigma)

		#def k(x,y):
		#	return np.exp(-1*(x-y).dot((x-y).T)/(2*self._sigma))
		self._Kmatrix = C
		return self._Kmatrix

	def Kernel_arr(self,Xtest):
		
		"""
		For prediction: 
		input  : x_test, a np.array en shape (m,p)
		output np.array, on shape (n,m)
		"""

		Y = np.asarray(Xtest)
		X = self._x
		n = X.shape[0]
		m = Y.shape[0]
		A = np.linalg.norm(X,axis = 1)[:,np.newaxis].dot(np.ones(shape = (1,m)))**2
		B = (np.linalg.norm(Y,axis = 1)[:,np.newaxis].dot(np.ones(shape = (1,n))).T)**2
		C = 2*X.dot(Y.T)
		ytemp = np.exp((C-A-B)/(2*self._sigma))
		return ytemp






class KernelRidge:

	def __init__(self, lmb = 0.01, kernel = None, sigma = 1.0):
	    	
		"""
		:Parameters:
		lmb : float
			The tuning parametre of ridge penalty 
		kernel : string
			Kernel type, can be 'linear' or 'rbf'
		sigma : the main parameter of gaussien kernel, only used when kernel = 'rbf'
		"""

		if kernel == None:
			self._kernel = linear_kernel()

		if kernel == 'linear':
			self._kernel = linear_kernel()

		if kernel == 'rbf':
			self._kernel = rbf_kernel(sigma = sigma)
			self._sigma = sigma
			
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

		self._alpha = np.linalg.solve(K_arr+self._lmb*n*np.eye(n),y_arr)
	                
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


	for i in range(1):
		x_train, x_test, y_train, y_test = train_test_split(xtr0, ytr0, test_size=0.2)
		clf = KernelRidge(lmb = 0.4, kernel = 'rbf', sigma = 1.0)
		
		start = time.time()
		clf.fit(x_train,y_train)
		print "fit time cost",time.time()-start

		start2 = time.time()
		clf.predict(x_test)
		print "predict time cost",time.time()-start2



	




