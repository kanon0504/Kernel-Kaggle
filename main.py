import numpy as np
from load_data import *
from combined_classifier import combined_classifier
from sklearn.cross_validation import train_test_split
from sklearn import svm
from KernelRidge import *
from kNN import kNN, verify
from cross_val_score_kNN import cross_val_score_kNN

xtr, ytr = load_data()
xte = load_data_test()
xte = np.asarray(flatten(xte))
xtr = np.asarray(flatten(xtr))
ytr = np.asarray([int(i) for i in ytr])


def cross_val_score(estimitor, xtr,ytr):
	xy = np.concatenate((xtr,ytr[:,np.newaxis]),axis=1)
	np.random.shuffle(xy)
	def get_split10_set(xy,i):
		'''
		@parameters:
		i : int
		the ith splited set 
		'''
		temp_te = xy[i*500:(i+1)*500]
		temp_tr = np.concatenate((xy[:i*500], xy[(i+1)*500:]),axis = 0)
		x_train = temp_tr[:,:784]
		y_train = temp_tr[:,784:]
		x_test = temp_te[:,:784]
		y_test = temp_te[:,784:]
		return x_train,x_test,y_train,y_test

	score = []
	for i in range(10):
		x_train,x_test,y_train,y_test = get_split10_set(xy,i)
		estimitor.fit(x_train,y_train)
		wtf = clf_combined.score(x_test, y_test)
		print wtf
		score.append(wtf)

	return np.mean(np.asarray(score))

'''
### parameters tuning (gaussian kernel + svm)
#find the best lamda and sigma with xtr and ytr by cross validation
clf = KernelRidge(lmb=0.05, kernel = 'rbf', sigma=1)
clf_combined = combined_classifier(clf)
wtf = cross_val_score(clf_combined,xtr,ytr)
print wtf

### final predict by svm
clf = KernelRidge(lmb=0.7, kernel = 'rbf', sigma=1)
clf_combined = combined_classifier(clf)
clf_combined.fit(xtr, ytr)
ypredict = clf_combined.predict(xte)
'''


'''
### parameters tuning (gaussian kernel + knn)
#find the best k and sigma with xtr and ytr by cross validation
xtr1, xtr2, ytr1, ytr2 = train_test_split(xtr, ytr, test_size=0.2)
for i in [1,3,5,7,9,11,13]:
	for j in np.linspace(1, 1.7, num = 8).tolist():
		scores = cross_val_score_kNN(xtr1, ytr1, i, j, 3) # cross validation with 3 folds
		print 'k:',i
		print 'sigma:',j
		print scores
### try model on xtr2 and ytr2 
ypredict = kNN(1, xtr1, ytr1, xtr2, 1.6)
scores = verify(ypredict, ytr2)
print scores
'''

### final predict by kNN
ypredict = kNN(5, xtr, ytr, xte,1.7)


f = open('Yte.csv','w')
f.write("Id,Prediction"+"\n")
for i in enumerate(ypredict):
	f.write(str(i[0]+1)+','+str(i[1])+"\n")
f.close()


