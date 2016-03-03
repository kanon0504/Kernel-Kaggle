import numpy as np
from load_data import load_data, plot, flatten
from combined_classifier import combined_classifier
from sklearn.cross_validation import train_test_split
from sklearn import svm

xtr, ytr = load_data()
xtr = flatten(xtr)

clf = svm.SVC(kernel='linear', C=1)

clf_combined = combined_classifier(clf)
x_train, x_test, y_train, y_test = train_test_split(xtr, ytr, test_size=0.2)
clf_combined.fit(x_train,y_train)
scores = clf_combined.score(x_test,y_test)
print scores