from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GridSearchCV

class naive_bayes:
    
    def  __init__(self,grid_search):
        
        self.grid_search = grid_search
        self.model = GaussianNB()
        self.parameters = {'var_smoothing':[1e-08, 1e-10]}
        self.CV = 5
        self.clf = GridSearchCV(self.model, self.parameters,
        							n_jobs = 4,
        							cv = self.CV)

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