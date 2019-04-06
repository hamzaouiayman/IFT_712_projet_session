from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV


class MLP:

	def __init__(self, grid_search):

		self.grid_search = grid_search
		self.model = MLPClassifier(solver='adam', alpha=1e-5, activation='relu', hidden_layer_sizes=(300, 200, 150, 100, 50),learning_rate='invscaling', random_state=0)
		self.parameters = {'hidden_layer_sizes': [(100,)], 'activation': ['logistic', 'relu'], 'solver': ['sgd'],
						   'alpha': ['0.0001'], 'max_iter': ['200']}
		self.CV = 5
		self.clf = GridSearchCV(self.model, self.parameters,
								n_jobs=4,
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
			score = self.clf.score(X_train, t_train)
		else:
			score = self.model.score(X_train, t_train)

		return score
