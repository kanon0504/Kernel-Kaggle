from scipy.optimize import fmin_l_bfgs_b
from scipy.optimize import minimize
import numpy as np
from load_data import load_data, plot, flatten
from sklearn.cross_validation import train_test_split


class linear_kernel:

    def kernel_create(self, x):
    	"""
    	create the kernel matrix
    	"""
        self._x = x
        self._Kmatrix = x.dot(x.T)
        return self._Kmatrix
    
    def Kernel_arr(self,x_arr):
		"""
		For prediction: 
		input an obsevation x
		output a n dimension vector K(x_i,x), i = 1,2,...,n
		"""
		return np.asarray([self._x[i].dot(x_arr) for i in range(self._x.shape[0])])




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


		def k(x,y):
			return np.exp(-1*(x-y).dot((x-y).T)/(2*self._sigma))


		self._Kmatrix = np.asarray([k(x[i],x[j]) for i in range(n) for j in range(n)]).reshape(n,n)


		return self._Kmatrix

	def Kernel_arr(self,x_arr):
		
		"""
		For prediction: 
		input  : an obsevation x, a np.array en shape (p,)
		output : a n dimension vector K(x_i,x), i = 1,2,...,n
		"""

		def k(x,y):
			return np.exp(-1*(x-y).dot((x-y).T)/(2*self._sigma))

		return np.asarray([k(self._x[i],x_arr) for i in range(self._x.shape[0])])





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
			
		if lmb < 0:
		    raise ValueError("lmb must be >= 0")

		self._lmb = float(lmb)        
		self._alpha = None

  
                                
    def fit(self, X, y):
    	
		"""
		To fit the model:
		input:  X_train, a np.array on shape (n,p)
			y_traiin, a np.array on shape (n,)
		"""

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
		Xtest = np.asarray(Xtest)
		p = [np.dot(self._alpha, self._kernel.Kernel_arr(Xtest[i])) for i in range(Xtest.shape[0])]
		return np.asarray(p)
    
    def score(self,Xtest,Ytest):
    	
		"""
		To fit the model:
		input:  X_test, a np.array on shape (m,p)
			y_train, a np.array on shape (m,)
		"""
		Xtest = np.asarray(Xtest)
		Ytest = np.asarray(Ytest)
		p = [np.dot(self._alpha, self._kernel.Kernel_arr(Xtest[i])) for i in range(Xtest.shape[0])]
		p = np.asarray([round(i) for i in  p])
		return float(sum(p == Ytest))/Xtest.shape[0]

	

if __name__ == '__main__':
	
	xtr0, ytr0 = load_data()
	
	ytr0 = np.asarray([int(i) for i in ytr0])
	xtr0 = flatten(xtr0)
	xtr0 = np.asarray(xtr0).reshape(len(xtr0),28*28)


	for i in range(1):
		x_train, x_test, y_train, y_test = train_test_split(xtr0, ytr0, test_size=0.2)
		clf = KernelRidge()
		clf.fit(x_train,y_train)
		print clf.score(x_test,y_test)




