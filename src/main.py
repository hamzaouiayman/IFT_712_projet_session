import pandas as pd
import json
import sys

from pandas.io.json import json_normalize

import processing.traitement_donnees as TD
import modeling.MLP as MLP
import modeling.SVM as SVM
import modeling.KNN as KNN
import modeling.RandomForest as RF
import modeling.regression_logistique as RL
import modeling.AdaBoost as AB

def main():

	if len(sys.argv) < 2:
		usage = "\n\n\t type_model: svm, knn, adaBoost, mlp, randomForest, regressionLogistique\
	        \n\t Grid_Search: 0:False 1:True\n"
		print(usage)
		return

	typeModel = sys.argv[1]
	grid_search = int(sys.argv[2])

	grid_search = True if (grid_search == 1) else False

	print("lecture des donnees")
	file_train = json.load(open("../Data/Raw/train.json"))
	file_test = json.load(open("../Data/Raw/test.json"))

	print("transformation des donnees")
	df_train = json_normalize(file_train)
	df_test = json_normalize(file_test)

	td = TD.TraitementDonnees(df_train,df_test)
	X_train,X_test = td.tfidf_transform()
	
	y_train = td.label_train_transform()

	#choix du model d'apprentissage
	print("entrainement")
	if (typeModel == 'knn') :
		model = KNN.KNN(grid_search)
	elif (typeModel == 'mlp'):
		model = MLP.MLP(grid_search)
	elif (typeModel == 'adaBoost'):
		model = AB.AdaBoost(grid_search)
	elif (typeModel == 'randomForest'):
		model = RF.RandomForest(grid_search)
	elif (typeModel == 'regressionLogistique'):
		model = RL.regression_logistique(grid_search)
	elif (typeModel == 'svm'):
		model = SVM.SVM(grid_search)

	#entrainement
	model.train(X_train,y_train)
	print("train accuracy",model.score(X_train,y_train))

	if (grid_search==True):
		model.print_cv_results()

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
