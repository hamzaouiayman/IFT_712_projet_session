from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV


class RandomForest:

	def  __init__(self):

		self.model = RandomForestClassifier(
							n_estimators=100, 
							random_state=0)
		self.parameters = {'n_estimators':[20, 30, 40, 50, 60, 100], 'max_depth':[10, 20, 40,50,80]}
		self.CV = 5
		self.clf = GridSearchCV(self.model, self.parameters,
									n_jobs = 4,
									cv = self.CV)


	def train(self, X_train, t_train, grid_serach):

		if (grid_serach == True):
			self.clf.fit(X_train, t_train)
			print(self.clf.cv_results_)
		else:
			self.model.fit(X_train, t_train)
			print(self.model.classes_)


	def predict(self, X_test, grid_serach):

		if (grid_serach == True):
			y_predict = self.clf.predict(X_test)
		else:
			y_predict = self.model.predict(X_test)

		return y_predict
