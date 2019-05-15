# -*- coding: utf-8 -*-
#Execute this file to run results of 3-part 
from LR3part import LR3partC
from SVM3part import SVM3partC
from RF3part import RF3partC 

models= [LR3partC(),SVM3partC(),RF3partC()]

for model in models:
    model_name = model.__class__.__name__
    list3part=model.execute3part()
    
    print("Accuracy",model_name,list3part)
          
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

    
N=3
metrics_values_lr=[0.86,0.79,0.54]
metrics_values_rf=[0.86,0.78,0.57]
metrics_values_svm=[0.875,0.822,0.5]
metrics_values_3part=[0.64,0.65,0.64]

ind = np.arange(N)  # the x locations for the groups
width = 0.25   

fig, ax = plt.subplots()
rects1 = ax.bar(ind,metrics_values_lr, width, color='r')
rects2 = ax.bar(ind+width, metrics_values_rf, width, color='b')
rects3 = ax.bar(ind+width*2, metrics_values_svm, width, color='g')


ax.legend((rects1[0], rects2[0],rects3[0]), ('Logistic', 'Random','SVM'),loc='center left',bbox_to_anchor=(1, 0.5))

ax.set_title('Accuracy-3 part Light-dataset')
ax.set_xticks(ind + width / 2)
ax.set_xticklabels(('Fact vs Expl','L-F vs L-EF','L-E vs L-EE'))
plt.show()

