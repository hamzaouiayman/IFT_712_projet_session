import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV


class MLP:

	def __init__(self, grid_search):

		self.grid_search = grid_search
		self.model = MLPClassifier(solver='adam', alpha=1e-5, activation='relu', hidden_layer_sizes=(300, 200, 150, 100, 50),learning_rate='invscaling', random_state=0)
		self.parameters = {'hidden_layer_sizes': [(300, 200, 150, 100, 50)], 'activation': ['logistic', 'relu','tanh'], 'solver': ['adam'],'learning_rate':['invscaling'],
						   'alpha': [1e-5]}
		self.CV = 5
		self.clf = GridSearchCV(self.model, self.parameters,
								verbose=2,
								n_jobs=5,
								cv=self.CV)

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
			print('meilleures parameteres :',self.clf.best_params_)
			score = self.clf.score(X_train, t_train)
		else:
			score = self.model.score(X_train, t_train)

		return score

	def print_cv_results(self):

		CV_result = pd.DataFrame(self.clf.cv_results_)
		CV_result.to_csv('../Data/CVresults/MLP_CV_result.csv')
