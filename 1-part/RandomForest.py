# -*- coding: utf-8 -*-
from ModelClass import Model
from sklearn.model_selection import KFold
from sklearn import metrics
import numpy as np
from sklearn.ensemble import RandomForestClassifier

class RandomForestC(Model):
    def __init__(self):
        print("Inside constructor")
    
    def GetPredictions(self,X_train,y_train,X_test):
        classifierRF = RandomForestClassifier(n_estimators=100,random_state=15325)
        classifierRF.fit(X_train,y_train)
        y_pred=classifierRF.predict(X_test)
        return y_pred
 
    def GetCrossValidation(self,X_train_dtm,y):
        list_accuracy=[]
        kf=KFold(n_splits=10)
        for train_index, test_index in kf.split(X_train_dtm):
            X_train, X_test = X_train_dtm[train_index], X_train_dtm[test_index]
            y_train, y_test = y[train_index], y[test_index]
            
            classifierRF = RandomForestClassifier(n_estimators=100,random_state=15325)
            classifierRF.fit(X_train,y_train)
    
            y_pred_class = classifierRF.predict(X_test)
            accuracy =metrics.accuracy_score(y_test, y_pred_class)
            list_accuracy.append(accuracy)
            average=np.average(list_accuracy)
            return average
