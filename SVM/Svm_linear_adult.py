#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  8 13:42:49 2020

@author: pranay
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
#column names has been updated for easy use
col_names = ["age","workclass","fnlwgt","education","education-num","marital_status","occupation","relationship","race","sex","capital_gain","capital_loss","hours-per-week","native_country","salary"]

#loading the dataset
dataset = pd.read_csv('adult.data',header=None,names = col_names)

dataset.dtypes

workclass_changed = {' Private':1, ' Self-emp-not-inc':2, ' Self-emp-inc':3, ' Federal-gov':4, ' Local-gov':5, ' State-gov':6, ' Without-pay':7, ' Never-worked':8," ?":9}
dataset.workclass = [workclass_changed[item] for item in dataset.workclass] 

education_changed = {' Bachelors':1, ' Some-college':2,' 11th':3,' HS-grad':4,' Prof-school':5,' Assoc-acdm':6,' Assoc-voc':7,' 9th':8,' 7th-8th':9,' 12th':10,' Masters':11,' 1st-4th':12,' 10th':13,' Doctorate':14,' 5th-6th':15,' Preschool':16}
dataset.education = [education_changed[item] for item in dataset.education] 

marital_changed = {' Married-civ-spouse':1,' Divorced':2,' Never-married':3,' Separated':4,' Widowed':5,' Married-spouse-absent':6,' Married-AF-spouse':7}
dataset.marital_status = [marital_changed[item] for item in dataset.marital_status] 

occupation_changed = {' Tech-support':1,' Craft-repair':2,' Other-service':3,' Sales':4,' Exec-managerial':5,' Prof-specialty':6,' Handlers-cleaners':7,' Machine-op-inspct':8,' Adm-clerical':9,' Farming-fishing':10,' Transport-moving':11,' Priv-house-serv':12,' Protective-serv':13,' Armed-Forces':14,' ?':15}
dataset.occupation = [occupation_changed[item] for item in dataset.occupation] 

relation_changed = {' Wife':1,' Own-child':2,' Husband':4,' Not-in-family':5,' Other-relative':6,' Unmarried':7 }
dataset.relationship = [relation_changed[item] for item in dataset.relationship] 

race_changed ={' race':1,' White':2,' Asian-Pac-Islander':3,' Amer-Indian-Eskimo':4,' Other':5,' Black':7}
dataset.race = [race_changed[item] for item in dataset.race]

sex_changed ={' Female':1,' Male':2}
dataset.sex = [sex_changed[item] for item in dataset.sex]

native_country_changed = {' United-States':1,' Cambodia':2,' England':3,' Puerto-Rico':4,' Canada':5,' Germany':6, 
' Outlying-US(Guam-USVI-etc)':7,' India':8,' Japan':9,' Greece':10,' South':11,' China':12,' Cuba':13,' Iran':14,' Honduras':15,' Philippines':16,
 ' Italy':17,' Poland':18,' Jamaica':19,' Vietnam':20,' Mexico':21,' Portugal':22,' Ireland':23,' France':24,' Dominican-Republic':25,' Laos':26,' Ecuador':27,
' Taiwan':28,' Haiti':29,' Columbia':30,' Hungary':31,' Guatemala':32,' Nicaragua':33,' Scotland':34,' Thailand':35,' Yugoslavia':36,' El-Salvador':37, 
' Trinadad&Tobago':38,' Peru':39,' Hong':40,' Holand-Netherlands':41,' ?':42}
dataset.native_country = [native_country_changed[item] for item in dataset.native_country]
#dataset attributes used for training and testing

attributes = ["occupation","relationship","race","sex","workclass","fnlwgt"]


x_data  = dataset[attributes]

y_data = dataset.capital_gain


X_train, X_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.3, random_state=1)

svclassifier = SVC(kernel='linear')

svclassifier.fit(X_train, y_train)

y_pred = svclassifier.predict(X_test)


print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))

#In SVM LINEAR set Adult with choosen attributes shows accuracy of 70% . Size of the dataset is 39,74,305  bytes
