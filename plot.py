import matplotlib.pyplot as plt

def plot(xtr,ytr):
	plt.figure(figsize=(14, 10))
	n_rows, n_cols = 4, 8
	for k in range(n_rows * n_cols):
	    plt.subplot(n_rows, n_cols, k + 1)
	    plt.imshow(xtr[k], cmap=plt.cm.gray, interpolation='none')
	    plt.xticks(())
	    plt.yticks(())    
	    plt.title(ytr[k], size=10)
	plt.show()