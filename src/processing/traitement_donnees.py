import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder

class TraitementDonnees:

	def __init__(self, df_train, df_test):
		self.df_train = df_train
		self.df_test = df_test
		self.tfidf = TfidfVectorizer(binary=True)
		self.LE = LabelEncoder()

	def tfidf_transform(self):
		#transformer les données en utilisant la methode tfidf
		
		#tokenization 
		self.df_train['ingredients']  = self.df_train['ingredients'].apply(lambda x : ' '.join([y.replace(' ','') for y in x]))
		self.df_test['ingredients']  = self.df_test['ingredients'].apply(lambda x : ' '.join([y.replace(' ','') for y in x]))

		#transformation tfidf
		X_train = self.tfidf.fit_transform(self.df_train['ingredients'])
		X_test = self.tfidf.transform(self.df_test['ingredients'])

		return X_train,X_test

	def label_train_transform(self):
		#transformer le vecteur cible 

		y_train  = self.df_train['cuisine']
		y = self.LE.fit_transform(y_train)
		return y

	def label_test_inverseTransform(self,labels):
		y_test = self.LE.inverse_transform(labels)
		return y_test

