import numpy as np
from load_data import *
from combined_classifier import combined_classifier
from sklearn.cross_validation import train_test_split
from sklearn import svm
from KernelRidge import *
from kNN import kNN
#from sklearn.kernel_ridge import KernelRidge
#from sklearn.grid_search import GridSearchCV

xtr, ytr = load_data()
xte = load_data_test()

xte = flatten(xte)
xtr = flatten(xtr)



### parameter tuning (gaussian kernel + svm)
#xtr1, xtr2, ytr1, ytr2 = train_test_split(xtr, ytr, test_size=0.2)
###find the best lmd and sigma with xtr1 and ytr1
#for i in [0.3,0.5,0.7]:
#	for j in [0.9,1.0,1.1,1.2]:
#		clf = KernelRidge(lmb=i, kernel = 'rbf', sigma=j)
#		clf_combined = combined_classifier(clf)
#		x_train, x_test, y_train, y_test = train_test_split(xtr1, ytr1, test_size=0.2)
#		clf_combined.fit(x_train,y_train)
#		scores = clf_combined.score(x_test,y_test)
#		print 'lmd:',i
#		print 'sigma:',j
#		print scores
### try model on xtr2 and ytr2 
#clf = KernelRidge(lmb=0.5, kernel = 'rbf', sigma=1)
#clf_combined = combined_classifier(clf)
#clf_combined.fit(xtr1,ytr1)
#scores = clf_combined.score(xtr2,ytr2)
#print scores



clf = KernelRidge(lmb=0.5, kernel = 'rbf', sigma=1)
clf_combined = combined_classifier(clf)
clf_combined.fit(xtr,ytr)
ypredict = clf_combined.predict(xte)



### parameter tuning (gaussian kernel + knn)
#xtr1, xtr2, ytr1, ytr2 = train_test_split(xtr, ytr, test_size=0.2)
#find the best k and sigma with xtr1 and ytr1
#for i in [1,3,5,7,9,11,13]:
#	for j in np.linspace(0.8, 1.2, num = 5).tolist():
#		x_train, x_test, y_train, y_test = train_test_split(xtr1, ytr1, test_size=0.2)
#		ypredict = kNN(i, x_train, y_train, x_test, j)
#		score = 0.
#		for q in range(len(y_test)):
#			if ypredict[q] == y_test[q]:
#				score += 1.
#		print 'k:',i
#		print 'sigma:',j
#		print score/len(y_test)



f = open('Yte.csv','w')
f.write("Id,Prediction"+"\n")
for i in enumerate(ypredict):
	f.write(str(i[0]+1)+','+str(i[1])+"\n")
f.close()


