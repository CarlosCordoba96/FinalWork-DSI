# -*- coding: utf-8 -*-
"""
Created on Sun May 12 12:42:37 2019

@author: Carlos
"""
# Libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.metrics import confusion_matrix
from sklearn.metrics import recall_score
from sklearn.utils import resample
import matplotlib.pyplot as plt
import itertools
import seaborn as sns
import pandas as pd


def balancear(dataframeX,dataframeY):
    dataframeX['mode_main']=dataframeY
    print(dataframeX['mode_main'].value_counts())
    
    
##Parametrizar las variables no numéricas
def male(x):
    if x=='yes':
        return 1
    if x=='no':
        return 0
 
def ethnicity(x):
    if x=='native':
        return 0
    if x=='western':
        return 1
    else:
        return 2

    #CARGAMOS LOS DATOS Y REALIZAMOS UN PREPROCESO EN LOS DATOS
df = pd.read_csv("nts_data.csv")
df['male']=df['male'].apply(male)
df['ethnicity']=df['ethnicity'].apply(ethnicity)
df['mode_main']=df['mode_main'].map({'walk':0,'car':1,'bike':2,'pt':3})
df['education']=df['education'].map({'lower':0,'middle':1,'higher':2})
df['income']=df['income'].map({'less20':0,'20to40':1,'more40':2})
df['license']=df['license'].map({'yes':1,'no':0})
df['weekend']=df['weekend'].map({'yes':1,'no':0})

print(df['mode_main'].value_counts())

#COJEMOS 100.000 DATOS DE MUESTRA
datos=df.sample(n=100000)
datos=datos.reset_index(drop=True)

datosy=datos['mode_main']
datos=datos.drop(columns='mode_main',axis=0)
#                   HITO 1
model=RandomForestClassifier(n_estimators =450,min_samples_split=3,n_jobs=-1)

"""
from sklearn.model_selection import train_test_split


X_train,X_test,y_train,y_test = train_test_split(datos,datosy,test_size=0.5, random_state=0)
model.fit(X_train,y_train)

from sklearn.metrics import accuracy_score,recall_score,confusion_matrix
y_pred = model.predict(X_test)
print(len(X_test))
print(accuracy_score(y_test,y_pred))
print(confusion_matrix(y_test,y_pred))

y_pred=cross_val_score(model,datos,datosy,cv=10,n_jobs=-1)
print(y_pred)
print(np.mean(y_pred))
"""

#KFOLD 10 CROSS-VALIDATION
X_trains=[]
X_tests=[]
y_trains=[]
y_tests=[]
scores = []
sensitivity=[]
from sklearn.model_selection import KFold # import KFold
kf = KFold(n_splits=10)
for train_index, test_index in kf.split(datos):
    X_train, X_test = datos.iloc[train_index], datos.iloc[test_index]
    y_train, y_test = datosy.iloc[train_index], datosy.iloc[test_index]
    n=int((X_train.shape[0]*(9/10))/4)
    X_train['mode_main']=y_train
    n=int((X_train.shape[0]*(9/10))/4)

    sub0=X_train.loc[X_train['mode_main']==0]
    sub0=resample(sub0,n_samples=n)
    
    sub1=X_train.loc[X_train['mode_main']==1]
    sub1=resample(sub1,n_samples=n)
    
    sub2=X_train.loc[X_train['mode_main']==2]
    sub2=resample(sub2,n_samples=n)
    
    sub3=X_train.loc[X_train['mode_main']==3]
    sub3=resample(sub3,n_samples=n)
    
    X_trained=sub0.append([sub1,sub2,sub3])


    y_train=X_trained['mode_main']
    X_train=X_trained.drop(columns='mode_main',axis=0)

    X_trains.append(X_train)
    X_tests.append(X_test)
    y_trains.append(y_train)
    y_tests.append(y_test)
    model.fit(X_train,y_train)
    scores.append(model.score(X_test, y_test))
    predicted=model.predict(X_test)
    average_precision = recall_score(y_test, predicted,average='macro')
    sensitivity.append(average_precision)
    
    print(scores)
    
print("Media de los valores: {}".format(np.mean(scores)))
print("Media de la sensitividad: {}".format(np.mean(sensitivity)))
print(model.feature_importances_)



#                           HITO 2
#importances = model.feature_importances_
#std = np.std([tree.feature_importances_ for tree in model.estimators_],
#             axis=0)
#indices = np.argsort(importances)[::-1]
#
## Print the feature ranking
#print("Feature ranking:")
#
#for f in range(X_train.shape[1]):
#    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))
#
#
## Plot the feature importances of the forest
#plt.figure()
#plt.title("Feature importances")
#plt.bar(range(len(model.feature_importances_)), importances,
#       color="r", yerr=std[indices], align="center")
#plt.xticks(range(X_train.shape[1]), list(X_train.columns.values), rotation='vertical')
#plt.xlim([-1, X_train.shape[1]])
#plt.show()
#print(list(X_train.columns.values))
