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

#COGEMOS 100.000 DATOS DE MUESTRA
datos=df.sample(n=100000)
datos=datos.reset_index(drop=True)

datosy=datos['mode_main']
datos=datos.drop(columns='mode_main',axis=0)



#                   HITO 1
#Se crea el modelo random forest con los parámetros especificados en el artículo, 450 árboles
#tres variables seleccionadas aleatoriamente para cada uno de los nodos de los árboles
model=RandomForestClassifier(n_estimators =450,min_samples_split=3,n_jobs=-1)


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
    n=int((X_train.shape[0]*(9/10))/4)#N para calcular nº de registros de cada tipo
    X_train['mode_main']=y_train
    #se junra la columna que queremos predecir y se filtran los datos por cada tipo
    #de viaje realizado y se realiza un sampleo de N y finalmente se juntan de nuevo
    #en un mismo dataset
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
    #Se guardan los conjuntos de datos para utilizarlos en posteriores hitos
    X_trains.append(X_train)
    X_tests.append(X_test)
    y_trains.append(y_train)
    y_tests.append(y_test)
    #Se cuadran los datos y se crea el modelo y se calcula precisión y sensibilidad
    model.fit(X_train,y_train)
    scores.append(model.score(X_test, y_test))#la precision
    
    predicted=model.predict(X_test)
    average_precision = recall_score(y_test, predicted,average='macro')#la sensibilidad
    
    sensitivity.append(average_precision)

hito1_acc=np.mean(scores)
hito1_sensi=np.mean(sensitivity)
print("Media de los valores: {}".format(np.mean(scores)))
print("Media de la sensibilidad : {}".format(np.mean(sensitivity)))
print(model.feature_importances_)#importancia de las variables en el modelo


#                           HITO 2
columnas=['distance','density','age','male','ethnicity','education','income','cars','license',
          'bicycles','weekend','diversity','green','temp','precip','wind']


plt.figure()
plt.title("Importancia variables ")
plt.barh(range(len(model.feature_importances_)), model.feature_importances_,color="r", align="center")
plt.yticks(range(len(model.feature_importances_)), columnas, rotation='horizontal')
plt.show()

hito2_scores=[]
hito2_sens=[]

acc2_scores=[]

for variable in columnas:#para cada variable
    for i in range(0,10):#en el 10-fold-cross-validatioon
        XNewTest=X_tests[i].copy()
        XNewTest[variable]=np.random.permutation(XNewTest[variable])#permutacion
        model.fit(X_trains[i],y_trains[i])
        hito2_scores.append(model.score(XNewTest, y_tests[i]))
        predicted=model.predict(XNewTest)
        average_precision = recall_score(y_tests[i], predicted,average='macro')
        hito2_sens.append(average_precision)
    #print("\tLA VARIABLE <<{}>>".format(variable))
    #print("Media de los valores: {}".format(np.mean(hito2_scores)))
    #print("Media de la sensibilidad : {}".format(np.mean(hito2_sens)))
    #print("Diferencia de la precisión: {}".format(np.mean(hito2_scores)-hito1_acc))
    #print("Diferencia de la sensibilidad: {}\n".format(np.mean(hito2_sens)-hito1_sensi))
    acc2_scores.append(np.mean(hito2_scores)-hito1_acc)
        

plt.figure()
plt.title("Accuracy")
plt.barh(range(len(acc2_scores)), acc2_scores,color="r", align="center")
plt.yticks(range(len(acc2_scores)), columnas, rotation='horizontal')
plt.show()


modo_viaje=[0,1,2,3]
hito2_scores=[]
hito2_sens=[]

acc_scores=[]
scores_modo=[]
for modo in modo_viaje:#para cada tipo de viaje
    for variable in columnas:#para cada variable
        for i in range(0,10):#en el 10fold-cross-validation
            XNewTest=X_tests[i].copy()#copiamos los datos
            XNewTest['mode_main']=y_tests[i]#juntamos todo en el mismo dataset
            sub0new=XNewTest.loc[XNewTest[XNewTest.mode_main==modo].index]#cogemos solo los registros que correspondan al modo de viaje
            XNewTest=XNewTest.drop(XNewTest[XNewTest.mode_main==modo].index)
            sub0new[variable]=np.random.permutation(sub0new[variable])#hacemos la permutacion
            XNewTest=XNewTest.append(sub0new)#lo juntamos
            y_tests[i]=XNewTest['mode_main']#actualizamos la y
            XNewTest=XNewTest.drop(columns='mode_main',axis=0)#eliminamos columnas
            
            model.fit(X_trains[i],y_trains[i])#entrenamos
            hito2_scores.append(model.score(XNewTest, y_tests[i]))
            predicted=model.predict(XNewTest)
            average_precision = recall_score(y_tests[i], predicted,average='macro')
            hito2_sens.append(average_precision)
#        print("\t EL MODO DE VIAJE:{}".format(modo) )
#        print("\tLA VARIABLE <<{}>>".format(variable))
#        print("Media de los valores: {}".format(np.mean(hito2_scores)))
#        print("Media de la sensibilidad : {}".format(np.mean(hito2_sens)))
#        print("Diferencia de la precisión: {}".format(np.mean(hito2_scores)-hito1_acc))
#        print("Diferencia de la sensibilidad: {}\n".format(np.mean(hito2_sens)-hito1_sensi))
#        acc_scores.append(np.mean(np.mean(hito2_sens)-hito1_sensi))
        print(acc_scores)
    plt.figure()
    plt.title("Relevancia {} ".format(modo))
    plt.barh(range(len(model.feature_importances_)), model.feature_importances_,color="r", align="center")
    plt.yticks(range(len(model.feature_importances_)), columnas, rotation='horizontal')
    plt.show() 
    scores_modo.append(acc_scores)
    print(scores_modo)
    acc_scores=[]

print(model.feature_importances_)#importancia de las variables en el modelo
    
for mode in modo_viaje:
    plt.figure()
    plt.title("Modo {} ".format(mode))
    plt.barh(range(len(scores_modo[mode])), scores_modo[mode],color="r", align="center")
    plt.yticks(range(len(scores_modo[mode])), columnas, rotation='horizontal')
    plt.show()

