from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import LogisticRegression


class AdaBoost:

	def __init__(self, grid_search):

		self.grid_search = grid_search
		self.model = LogisticRegression(C=5, class_weight=None, fit_intercept=True, multi_class='multinomial',
										solver='sag', tol=1e-05)
		self.estimators = 5
		self.clf = AdaBoostClassifier(self.model, n_estimators=3)

	def train(self, X_train, t_train):

		if (self.grid_search == True):
			self.clf.fit(X_train.toarray(), t_train)
		else:
			self.model.fit(X_train.toarray(), t_train)

	def predict(self, X_test):

		if (self.grid_search == True):
			y_predict = self.clf.predict(X_test.toarray())
		else:
			y_predict = self.model.predict(X_test.toarray())

		return y_predict

	def score(self, X_train, t_train):

		if (self.grid_search == True):
			score = self.clf.score(X_train.toarray(), t_train)
		else:
			score = self.model.score(X_train.toarray(), t_train)

		return score

	def print_cv_results(self):
			return
