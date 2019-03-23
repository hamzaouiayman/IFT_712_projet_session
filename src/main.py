import pandas as pd
import json
from pandas.io.json import json_normalize

import processing.traitement_donnees as TD

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



if __name__ == "__main__":
    main()
