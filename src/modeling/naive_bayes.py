from sklearn.naive_bayes import GaussianNB
#from sklearn.model_selection import GridSearchCV

class naive_bayes:
    
    def  __init__(self):
        
		self.model = GaussianNB()
        #self.parameters = {'var_smoothing':[1e-08, 1e-10]}        
        #self.CV = 5
		#self.clf = GridSearchCV(self.model, self.parameters,
		#							n_jobs = 4,
		#							cv = self.CV)



    def train(self, X_train, t_train):

		self.model.fit(X_train, t_train)

    def predict(self, X_test):

		y_predict = self.model.predict(X_test)

		return y_predict