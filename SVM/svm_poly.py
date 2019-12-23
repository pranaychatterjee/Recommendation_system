#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 23 17:53:54 2019

@author: pranay
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix



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

svclassifier = SVC(kernel='poly', degree=8)
svclassifier.fit(X_train, y_train)

y_pred = svclassifier.predict(X_test)

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

#Gausian Kernel

svclassifier = SVC(kernel='rbf')
svclassifier.fit(X_train, y_train)

y_pred = svclassifier.predict(X_test)

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

#Sigmoid Kernel

svclassifier = SVC(kernel='sigmoid')
svclassifier.fit(X_train, y_train)

y_pred = svclassifier.predict(X_test)

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
