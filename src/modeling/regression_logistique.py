from sklearn.linear_model import LogisticRegression
#from sklearn.model_selection import GridSearchCV

class regression_logistique:
    
    def  __init__(self):
        
		self.model = LogisticRegression()
        self.parameters = {'max_iter':[50, 200], 'solver':['newton-cg', 'sag', 'lbfgs'], 'multi_class':['multinomial'],
                           'tol':[1e-3, 1e-5], 'C':[1, 10], 'fit_intercept':[True]}        
        self.CV = 5
		self.clf = GridSearchCV(self.model, self.parameters,
									n_jobs = 4,
									cv = self.CV)



    def train(self, X_train, t_train):

		self.model.fit(X_train, t_train)

    def predict(self, X_test):

		y_predict = self.model.predict(X_test)

		return y_predict