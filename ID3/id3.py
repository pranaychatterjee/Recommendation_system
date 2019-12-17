#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 13 09:48:53 2019

@author: pranay
"""

from sklearn.tree import DecisionTreeClassifier
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

col_names = ["erythema",
      "scaling",
      "definite borders",
      "itching",
      "koebner phenomenon",
      "polygonal papules",
      "follicular papules",
      "oral mucosal involvement",
      "knee and elbow involvement",
     "scalp involvement",
     "family history",
     "melanin incontinence",
     "eosinophils in the infiltrate",
     "PNL infiltrate",
     "fibrosis of the papillary dermis",
     "exocytosis",
     "acanthosis",
     "hyperkeratosis",
     "parakeratosis",
     "clubbing of the rete ridges",
     "elongation of the rete ridges",
     "thinning of the suprapapillary epidermis",
     "spongiform pustule",
     "munro microabcess",
     "focal hypergranulosis",
     "disappearance of the granular layer",
     "vacuolisation and damage of basal layer",
     "spongiosis",
     "saw-tooth appearance of retes",
     "follicular horn plug",
     "perifollicular parakeratosis",
     "inflammatory monoluclear inflitrate",
     "?",
     "Age",
     "band-like infiltrate"
     ]

dataset = pd.read_csv("dermatology.data",header=None,names=col_names)

dataset

attributes =["erythema",
      "scaling",
      "definite borders",
      "itching",
      "koebner phenomenon",
      "polygonal papules",
      "follicular papules",
      "oral mucosal involvement"]

x_data = dataset[attributes]

y_data = dataset.Age
'''
train_features = dataset.iloc[:80,:-1]
test_features = dataset.iloc[80:,:-1]
train_targets = dataset.iloc[:80,-1]
test_targets = dataset.iloc[80:,-1]
'''

X_train, X_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.3, random_state=1)

tree = DecisionTreeClassifier(criterion = 'entropy').fit(X_train,y_train)

prediction = tree.predict(X_test)

print("The prediction accuracy is: ",tree.score(X_test,y_test)*100,"%")