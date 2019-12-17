#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 12 18:51:35 2019

@author: pranay
"""

import pandas as pd 
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation

#libraries for graphical view
from sklearn.tree import export_graphviz
from sklearn.externals.six import StringIO  
from IPython.display import Image  
import pydotplus

#column names has been updated for easy use
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

#loading the dataset
dataset = pd.read_csv('dermatology.data',header=None,names = col_names)


#dataset attributes used for training and testing
attributes = ["erythema",
      "scaling",
      "definite borders",
      "itching",
      "koebner phenomenon",
      "polygonal papules",
      "follicular papules",
      "oral mucosal involvement"]


x_data  = dataset[attributes]

y_data = dataset.hyperkeratosis


X_train, X_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.3, random_state=1)

# Create Decision Tree classifer object
clf = DecisionTreeClassifier()

# Train Decision Tree Classifer
clf = clf.fit(X_train,y_train)

#Predict the response for test dataset
y_pred = clf.predict(X_test)

# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

#graphical representation

dot_data = StringIO()
export_graphviz(clf, out_file=dot_data,  
                filled=True, rounded=True,
                special_characters=True,feature_names = attributes,class_names=None)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
graph.write_png('dermatology.png')
Image(graph.create_png())

