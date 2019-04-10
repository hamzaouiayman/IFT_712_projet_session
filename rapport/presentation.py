#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 23 16:31:24 2019

@author: julien

This page is used to create diagrams for the presentation
"""

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from collections import Counter
plt.style.use('ggplot')
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

train_df = pd.read_json('./Data/Raw/train.json')
test_df = pd.read_json('./Data/Raw/test.json')

train_df['seperated_ingredients'] = train_df['ingredients'].apply(','.join)
test_df['seperated_ingredients'] = test_df['ingredients'].apply(','.join)

print("Nombre maximum d'ingrédients dans un plat: ",train_df['ingredients'].str.len().max())
print("Nombre minimum d'ingrédients dans un plat: ",train_df['ingredients'].str.len().min())
print("Nombre de catégories de cuisine: {}".format(len(train_df.cuisine.unique())))
print (train_df.cuisine.unique())
# Get an histogram of the number of ingredient by recipe
plt.hist(train_df['ingredients'].str.len(),bins=max(train_df['ingredients'].str.len()),edgecolor='b')
plt.gcf().set_size_inches(14,7)
plt.title('Ingredients dans la distribution des plats')

"""
# Obtenir les informations sur la distribution des cuisines
# ie : Le nombre de cuisines Italienne ou Française
sns.countplot(y='cuisine', data=train_df)
plt.gcf().set_size_inches(12,8)
plt.title('Cuisine Distribution',size=15)
"""

# Get histogram of the ingredients most consumed per cuisine
train_df['for ngrams']=train_df['seperated_ingredients'].str.replace(',',' ')
f,ax=plt.subplots(2,2,figsize=(20,20))
def ingre_cusine(cuisine):
    frame=train_df[train_df['cuisine']==cuisine]
    common=list(nltk.bigrams(nltk.word_tokenize(" ".join(frame['for ngrams']))))
    return pd.DataFrame(Counter(common),index=['count']).T.sort_values('count',ascending=False)[:15]
ingre_cusine('mexican').plot.barh(ax=ax[0,0],width=0.9,color='#45ff45')
ax[0,0].set_title('Cuisine Mexicaine')
ingre_cusine('indian').plot.barh(ax=ax[0,1],width=0.9,color='#df6dfd')
ax[0,1].set_title('Cuisine Indienne')
ingre_cusine('italian').plot.barh(ax=ax[1,0],width=0.9,color='#fbca5f')
ax[1,0].set_title('Cuisine Italienne')
ingre_cusine('chinese').plot.barh(ax=ax[1,1],width=0.9,color='#ffff00')
ax[1,1].set_title('Cuisine Chinoise')
plt.subplots_adjust(wspace=0.5)