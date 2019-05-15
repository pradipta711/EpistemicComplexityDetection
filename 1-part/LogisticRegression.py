from ModelClass import Model
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn import metrics
import numpy as np


class LogisticRegressionC(Model):
    
    def __init__(self):
        print("Inside constructor LR")
    
    def GetPredictions(self,X_train,y_train,X_test):
        logisticRegr = LogisticRegression()
        logisticRegr.fit(X_train,y_train);
        y_pred=logisticRegr.predict(X_test)
        return y_pred
 
    def GetCrossValidation(self,X_train_dtm,y):
        list_accuracy=[]
        kf=KFold(n_splits=10)
        for train_index, test_index in kf.split(X_train_dtm):
            X_train, X_test = X_train_dtm[train_index], X_train_dtm[test_index]
            y_train, y_test = y[train_index], y[test_index]
            
            logisticRegr = LogisticRegression()
            logisticRegr.fit(X_train, y_train)
            y_pred_class = logisticRegr.predict(X_test)
            accuracy =metrics.accuracy_score(y_test, y_pred_class)
            list_accuracy.append(accuracy)
            average=np.average(list_accuracy)
            return average
