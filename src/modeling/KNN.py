import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV

class KNN:

	def  __init__(self,grid_search):

		self.grid_search = grid_search
		self.model = KNeighborsClassifier(
							n_neighbors=5,
							weights = 'distance')
		self.parameters = {'n_neighbors':[5], 'weights':['distance']}#,'uniform'
		self.CV = 5
		self.clf = GridSearchCV(self.model, self.parameters,
									n_jobs = 4,
									cv = self.CV)

	def train(self, X_train, t_train):

		if (self.grid_search == True):
			self.clf.fit(X_train, t_train)
		else:
			self.model.fit(X_train, t_train)


	def predict(self, X_test):

		if (self.grid_search == True):
			y_predict = self.clf.predict(X_test)
		else:
			y_predict = self.model.predict(X_test)

		return y_predict

	def score(self, X_train, t_train):
		if (self.grid_search == True):
			score = self.clf.score(X_train, t_train)
		else:
			score = self.model.score(X_train, t_train)

		return score

	def print_cv_results(self):

			CV_result = pd.DataFrame(self.clf.cv_results_)
			CV_result.to_csv('../Data/CVresults/KNN_CV_result.csv')