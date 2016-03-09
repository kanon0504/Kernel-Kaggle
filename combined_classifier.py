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

	def predict(self,xtr):
		score = []
		for i in self.clf_combined:			
			score.append(i.predict(np.array(xtr)))
		score = np.array(score).T
		opt = np.argmax(score, axis = 1)
		output = [self.class_list[i] for i in opt]
		return output
	# input: each data entry of the database
	# output: the prediction class in string type

	def score(self,xtr,ytr):
		score = 0.
		y = self.predict(xtr)
		for i in range(len(ytr)):
			if y[i] == ytr[i]:
				score += 1.
		return score/len(ytr)



