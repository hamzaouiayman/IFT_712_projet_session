from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV

class KNN:

	def  __init__(self):

		self.model = KNeighborsClassifier(
							n_neighbors=5,
							weights = 'distance')
		self.parameters = {'n_neighbors':[5, 7, 9, 10, 20], 'weights':['distance','uniform']}
		self.CV = 5
		self.clf = GridSearchCV(self.model, self.parameters,
									n_jobs = 4,
									cv = self.CV)


	def train(self, X_train, t_train, grid_serach):

		if (grid_serach == True):
			self.clf.fit(X_train, t_train)
		else:
			self.model.fit(X_train, t_train)


	def predict(self, X_test, grid_serach):

		if (grid_serach == True):
			y_predict = self.clf.predict(X_test)
		else:
			y_predict = self.model.predict(X_test)

		return y_predict