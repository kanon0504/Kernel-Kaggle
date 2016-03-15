import numpy as np 

class kpca(object):

	def __init__(self):
		self.feature_number = 40
		pass

	def fit_transform(self,x,feature_number):
		n = len(x)
		m = len(x[0])
		x = np.asarray(x)
		# mean = np.mean(x,axis = 1)
		# x_centered = x - (mean * np.ones(m*n).reshape(n,m)).T
		# # centralize x

		# c = np.dot(x_centered,x_centered.T) * (1./n)
		# # covariance matrix

		# u,s,v = np.linalg.svd(c)
		# return np.dot(x,u[:feature_number].T)

		k = np.dot(x,x.T)
		k_il = (np.mean(k,axis=0)*np.ones(n**2).reshape(n,n)).T/n
		k_kj = (np.mean(k,axis=1)*np.ones(n**2).reshape(n,n)).T/n
		k_kl = (np.mean(a)*np.ones(n**2).reshape(n,n))/(n**2)
		k_centered = k - k_il - k_kj + k_kl
		# centralize k

		u_k,_,_ = np.linalg.svd(k_centered)

		x_reduced = np.dot(k,u_k[:feature_number].T)

		return x_reduced
