# -*- coding: utf-8 -*-
"""
Created on Mon Jul  5 08:12:56 2021

@author: Vivek
"""

#Importing the libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

#Loading the dataset
forestfires = pd.read_csv("E://8sept//assigement//Support Vector Machines//forestfires.csv")
data=forestfires.describe()


#Dropping the month and day columns
forestfires.drop(["month","day"], axis=1,inplace=True)

#Setting the x(features) and y(target)
predictors = forestfires.iloc[:,0:28]
target = forestfires.iloc[:,28]

#Normalising the data as there is scale difference
def norm_func(i):
    x= (i-i.min())/(i.max()-i.min())
    return (x)

fires = norm_func(predictors)
from sklearn.model_selection import train_test_split

#train_test_split
x_train,x_test,y_train,y_test = train_test_split(predictors,target, test_size=0.3, random_state=0)

#Model building using SVM
from sklearn.svm import SVC
model_linear = SVC(kernel = "linear")
model_linear.fit(x_train,y_train)
pred_linear = model_linear.predict(x_test)
#Accuracy
np.mean(pred_linear==y_test)
#98%

#Kernel=poly
model_poly = SVC(kernel = "poly")
model_poly.fit(x_train,y_train)
pred_poly = model_poly.predict(x_test)
#Accuracy
np.mean(pred_poly==y_test)
#75.6%

#Kernel=rbf
model_rbf = SVC(kernel = 'rbf')
model_rbf.fit(x_train,y_train)
pred_rbf = model_rbf.predict(x_test)
#Accuracy
np.mean(pred_rbf==y_test)
#72.43

#Kernel=sigmoid
model_sigmoid = SVC(kernel ='sigmoid')
model_sigmoid.fit(x_train,y_train)
pred_sigmoid = model_sigmoid.predict(x_test)
#Accuracy
np.mean(pred_sigmoid==y_test)
#71.15






