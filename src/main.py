import pandas as pd
import json
from pandas.io.json import json_normalize

import processing.traitement_donnees as TD
import modeling.SVM as SVM

def main():

	print("lecture des données")
	file_train = json.load(open("../Data/Raw/train.json"))
	file_test = json.load(open("../Data/Raw/test.json"))

	print("transformation des données")
	df_train = json_normalize(file_train)
	df_test = json_normalize(file_test)

	td = TD.TraitementDonnees(df_train,df_test)
	X_train,X_test = td.tfidf_transform()
	
	y_train = td.label_train_transform()

	model = SVM.SVM()
	model.train(X_train,y_train,False)
	y_pred = model.predict(X_test,False)

	y_labels = td.label_test_inverseTransform(y_pred)
	print(y_labels)

	test_id = df_test['id']
	sub = pd.DataFrame({'id': test_id, 'cuisine': y_labels}, columns=['id', 'cuisine'])
	sub.to_csv('svm_output.csv', index=False)

if __name__ == "__main__":
    main()
