# -*- coding: utf-8 -*-
from ModelClass import Model
from LogisticRegression import LogisticRegressionC
from RandomForest import RandomForestC
from SVM import SVMC
import matplotlib.pyplot as plt
import numpy as np
models= [LogisticRegressionC(),RandomForestC(),SVMC()]

for model in models:
    model_name = model.__class__.__name__
    listTR=model.SetTrainingData()
    X_train=listTR[0]
    X_test=listTR[1]
    y_train=listTR[2]
    y_test=listTR[3]
    X=listTR[4]
    y=listTR[5]
    y_pred=model.GetPredictions(X_train,y_train,X_test)
    acc=model.GetMetrics(y_test, y_pred)
    print("Accuracy",model_name,acc)
    avg_accuracy=model.GetCrossValidation(X,y)
    print("Cross Validation Accuracy:",model_name,avg_accuracy)
     
#code for plotting the comparison results between the 3 models for the light dataset   
N=3
metrics_values_kfold=[0.71,0.67,0.72]
metrics_values=[0.69,0.68,0.695]
ind = np.arange(N)  # the x locations for the groups
width = 0.25   

fig, ax = plt.subplots()
rects1 = ax.bar(ind, metrics_values_kfold, width, color='r')
rects2 = ax.bar(ind+width, metrics_values, width, color='b')

ax.legend((rects1[0], rects2[0]), ('Metrics(KFold)', 'Metrics(Without KFold)'),loc='center left',bbox_to_anchor=(1, 0.5))

ax.set_title('Accuracy-1 part Light-dataset')
ax.set_xticks(ind + width / 2)
ax.set_xticklabels(('Logistic Regression','Random Forest','SVM'))
plt.show()
    
#code for plotting the comparison results between the 3 models for the soil dataset   
N=3
metrics_values_kfold=[0.47,0.41,0.529]
metrics_values=[0.42,0.48,0.6]
ind = np.arange(N)  # the x locations for the groups
width = 0.25   

fig, ax = plt.subplots()
rects1 = ax.bar(ind, metrics_values_kfold, width, color='g')
rects2 = ax.bar(ind+width, metrics_values, width, color='y')

ax.legend((rects1[0], rects2[0]), ('Metrics(KFold)', 'Metrics(Without KFold)'),loc='center left',bbox_to_anchor=(1, 0.5))

ax.set_title('Accuracy-1 part Soil-dataset')
ax.set_xticks(ind + width / 2)
ax.set_xticklabels(('Logistic Regression','Random Forest','SVM'))
plt.show()
      
   
       
    
    
    
    
    
    