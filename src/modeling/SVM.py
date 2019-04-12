import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.multiclass import OneVsRestClassifier


class SVM:

	def __init__(self, grid_search):

		self.grid_search = grid_search
		self.classifier = SVC()
		self.model = OneVsRestClassifier(self.classifier)
		self.parameters = {'kernel': ['poly', 'rbf', 'sigmoid'], 'C': [1, 10,50,100],'tol':[0.001,0.00001],'degree':[3,5,8],'gamma':[1]}
		self.CV = 5
		self.clf = GridSearchCV(self.classifier, self.parameters,
								n_jobs=6,
								verbose=2,
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
