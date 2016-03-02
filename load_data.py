import csv
import numpy as np 
import matplotlib.pyplot as plt


def load_data():
	xtr = []
	ytr = []
	
	with open('Xtr.csv','rb') as csvfile:
		reader = csv.reader(csvfile, delimiter = ',')
		for row in reader:
			row = np.array(row).reshape(28,28)
			xtr.append(list(row.astype(float)))

	with open('Ytr.csv','rb') as csvfile:
		reader = csv.DictReader(csvfile, delimiter = ',')
		for row in reader:
			ytr.append(row['Prediction'])

	return xtr, ytr


def flatten(xtr):
	for i in range(len(xtr)):
		xtr[i] = np.array(xtr[i]).flatten()

	return xtr


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