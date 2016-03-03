import numpy as np 
import copy


class combined_classifier(object):
	def __init__(self,object):
		self.clf = object
		self.clf_combined = []

	def fit(self,xtr,ytr):
		self.class_list = list(set(ytr))
		for item in self.class_list:
			clf_temp = copy.deepcopy(self.clf)
			y_temp = []
			for i in ytr:
				if i == item:
					y_temp.append(1)
				else:
					y_temp.append(0)
			clf_temp.fit(xtr,y_temp)
			self.clf_combined.append(clf_temp)

	def predict(self,x):
		score = []
		for i in self.clf_combined:
			score.append(i.predict(x.reshape(1,-1)))
		opt = score.index(max(score))
		return self.class_list[opt]

	def score(self,xtr,ytr):
		score = 0.
		for i in range(len(xtr)):
			if ytr[i] == self.predict(xtr[i]):
				score += 1.
		return score/len(ytr)


