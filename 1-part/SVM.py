# -*- coding: utf-8 -*-

from ModelClass import Model
from sklearn.model_selection import KFold
from sklearn import metrics
import numpy as np
from sklearn.svm import SVC


class SVMC(Model):
    
    def __init__(self):
        print("Inside constructor SVM")
    
    def GetPredictions(self,X_train,y_train,X_test):
        classifierSVC = SVC(kernel = 'linear', random_state = 0)
        classifierSVC.fit(X_train, y_train)
        y_pred=classifierSVC.predict(X_test)
        return y_pred
       
 
    def GetCrossValidation(self,X_train_dtm,y):
        list_accuracy=[]
        kf=KFold(n_splits=10)
        for train_index, test_index in kf.split(X_train_dtm):
            X_train, X_test = X_train_dtm[train_index], X_train_dtm[test_index]
            y_train, y_test = y[train_index], y[test_index]
            
            classifierSVC = SVC(kernel = 'linear', random_state = 0)
            classifierSVC.fit(X_train, y_train)
        
            y_pred_class = classifierSVC.predict(X_test)
            accuracy =metrics.accuracy_score(y_test, y_pred_class)
            list_accuracy.append(accuracy)
            average=np.average(list_accuracy)
            return average
 
