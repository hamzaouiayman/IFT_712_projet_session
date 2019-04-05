import pandas as pd
import json
from pandas.io.json import json_normalize

import processing.traitement_donnees as TD
import modeling.MLP as MLP
import modeling.SVM as SVM
import modeling.KNN as KNN
import modeling.RandomForest as RF
import modeling.regression_logistique as RL
import modeling.naive_bayes as NB

def main():

	print("lecture des donnees")
	file_train = json.load(open("../Data/Raw/train.json"))
	file_test = json.load(open("../Data/Raw/test.json"))

	print("transformation des donnees")
	df_train = json_normalize(file_train)
	df_test = json_normalize(file_test)

	td = TD.TraitementDonnees(df_train,df_test)
	X_train,X_test = td.tfidf_transform()
	
	y_train = td.label_train_transform()
	grid_search = False

	#choix du model d'apprentissage
	print("entrainement")
	typeModel = "randomForest"
	if (typeModel == 'knn') : 
		model = KNN.KNN(grid_search)
	elif (typeModel == 'mlp'):
		model = MLP.MLP(grid_search)
	elif (typeModel == 'naiveBayes'):
		model = NB.naive_bayes(grid_search)
	elif (typeModel == 'randomForest'):
		model = RF.RandomForest(grid_search)
	elif (typeModel == 'regressionLogistique'):
		model = RL.regression_logistique(grid_search)
	elif (typeModel == 'svm'):
		model = SVM.SVM(grid_search)

	#entrainement
	model.train(X_train,y_train)
	print("train accuracy",model.score(X_train,y_train))

	#prediction des données de test
	print("prediction")
	y_pred = model.predict(X_test)

	#transormer les classes en label
	y_labels = td.label_test_inverseTransform(y_pred)
	print(y_labels)

	#sauvegarde des resultats dans un fichiers csv
	test_id = df_test['id']
	sub = pd.DataFrame({'id': test_id, 'cuisine': y_labels}, columns=['id', 'cuisine'])
	sub.to_csv('../Data/results/'+typeModel+'_test_result.csv', index=False)

if __name__ == "__main__":
    main()
