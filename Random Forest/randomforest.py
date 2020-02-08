#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 17 12:11:33 2019

@author: pranay
"""

import pandas as pd 
import numpy as np
#Import Random Forest Model
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation

#libraries for graphical view
from sklearn.tree import export_graphviz
from sklearn.externals.six import StringIO  
from IPython.display import Image  
import pydotplus

from sklearn.datasets import make_blobs

import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

from sklearn import preprocessing

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

#dataset has been imported
dataset = pd.read_csv('dermatology.data',header=None,names = col_names)

#dataset attributes used for training and testing
attributes = ["erythema",
      "scaling","definite borders",
      "itching",
      "koebner phenomenon",
      "polygonal papules",
      "follicular papules",
      "oral mucosal involvement"
      ]

x_data  = dataset[attributes]

y_data = dataset.hyperkeratosis

X_train, X_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.3, random_state=1)

clf=RandomForestClassifier(n_estimators=100)

#Train the model using the training sets y_pred=clf.predict(X_test)
clf.fit(X_train,y_train)

y_pred=clf.predict(X_test)

print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

#In Random Forest data set Dermatology with choosen attributes shows accuracy of 58.18% . Size of the dataset is 25,964  bytes
