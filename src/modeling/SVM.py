import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.multiclass import OneVsRestClassifier


class SVM:

	def __init__(self, grid_search):

		self.grid_search = grid_search
		self.classifier = SVC(C=100,  # penalty parameter
							  kernel='rbf',  # kernel type, rbf working fine here
							  degree=3,  # default value
							  gamma=1,  # kernel coefficient
							  coef0=1,  # change to 1 from default value of 0.0
							  tol=0.001,  # stopping criterion tolerance
							  probability=False,  # no need to enable probability estimates
							  class_weight=None,  # all classes are treated equally
							  verbose=False  # print the logs
							  )
		self.model = OneVsRestClassifier(self.classifier)
		self.parameters = {'kernel': ['rbf', 'linear'], 'C': [1, 10]}
		self.CV = 5
		self.clf = GridSearchCV(self.classifier, self.parameters,
								n_jobs=6,
								cv=self.CV)

	def train(self, X_train, t_train):

		if (self.grid_search == True):
			self.clf.fit(X_train, t_train)
		else:
			self.model.fit(X_train, t_train)

	def predict(self, X_test):

		if (self.grid_search==True):
			y_predict = self.clf.predict(X_test)
		else:
			y_predict = self.model.predict(X_test)

		return y_predict

	def score(self, X_train, t_train):

		if (self.grid_search == True):
			print('meilleures parameteres :', self.clf.best_params_)
			score = self.clf.score(X_train, t_train)
		else:
			score = self.model.score(X_train, t_train)

		return score

	def print_cv_results(self):

		CV_result = pd.DataFrame(self.clf.cv_results_)
		CV_result.to_csv('../Data/CVresults/SVM_CV_result.csv')
