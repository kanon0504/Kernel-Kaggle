import csv
import numpy as np 

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