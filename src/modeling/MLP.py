from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV

class MLP:
    
    def  __init__(self):
        
        #MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
		self.model = SVC()
        self.parameters = {'hidden_layer_sizes':[(100,)], 'activation':['logistic', 'relu'], 'solver':['sgd'],
                                                 'alpha':['0.0001'], 'max_iter':['200']}        
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
