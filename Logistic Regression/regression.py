#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 17 10:43:27 2019

@author: pranay
"""

import pandas as pd 
import numpy as np
from sklearn.linear_model import LogisticRegression # Import Logistic Regression Classifier
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

#X, y = make_blobs(n_samples=100, centers=3, n_features=2)
# create and configure model
#model = LogisticRegression(solver='lbfgs')
# fit model
#model.fit(X, y)

X_train, X_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.3, random_state=1)

model = LogisticRegression(solver='lbfgs', multi_class='auto')

model.fit(X_train,y_train)

#
y_pred=model.predict(X_test)

cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
cnf_matrix

print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

#In Regression data set Dermatology with choosen attributes shows accuracy of 70.00% . Size of the dataset is 25,964 bytes


class_names=[2,3] # name  of classes
fig, ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)
# create heatmap
sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu" ,fmt='g')
ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.title('Confusion matrix', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')

