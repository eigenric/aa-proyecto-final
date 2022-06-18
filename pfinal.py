# -*- coding: utf-8 -*-
"""
Created on Sat Jun  4 09:11:30 2022

@author: Daniel Navarrete Martin
@author: Ricardo Ruiz Fernandez Alba
"""

# Tratamiento de datos
# ==============================================================================
import pandas as pd
import numpy as np
from collections import Counter
from sklearn.impute import SimpleImputer
# from sklearn.feature_selection import RFE
import matplotlib.pyplot as plt
# from sklearn.ensemble import RandomForestClassifier
from sklearn.experimental import enable_iterative_imputer
from imblearn.over_sampling import SMOTE
from sklearn.impute import IterativeImputer
from sklearn.linear_model import Ridge
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate
from sklearn.model_selection import cross_val_score
# import joblib
# from sklearn.model_selection import RandomizedSearchCV
# from sklearn.model_selection import GridSearchCV
# from sklearn.metrics import roc_curve,auc
from sklearn.metrics import confusion_matrix
# from sklearn.naive_bayes import GaussianNB
# from scipy.stats import uniform,randint
# from tqdm import tqdm
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.ensemble import RandomForestClassifier
# from xgboost import XGBClassifier
# from sklearn.preprocessing import MinMaxScaler
# from lightgbm import LGBMClassifier
from sklearn.metrics import f1_score
# from sklearn.metrics import precision_recall_curve
from imblearn.under_sampling import RandomUnderSampler
# from imblearn.pipeline import Pipeline
# from sklearn.linear_model import SGDClassifier
# from sklearn.ensemble import AdaBoostClassifier
# from prettytable import PrettyTable
# import pickle
import random

import warnings
warnings.filterwarnings("ignore")



#Lectura de datos


datos = pd.read_csv('datos/aps_failure_training_set.csv', skiprows=20, na_values=["na"])
datos['class'] = datos['class'].replace(['neg','pos'],[0,1])

# Distribución de la etiqueta class
# Neg -> 0
# Pos -> 1
sns.barplot(datos['class'].unique(),datos['class'].value_counts())
plt.title('Class Label Distribution')
plt.xlabel('Class Label')
plt.ylabel('Count')
plt.show()

print('Numero de clases positivas: ',datos['class'].value_counts()[1])
print('Numero de clases negativas: ',datos['class'].value_counts()[0])




#Quita las variables con varianza cero
def constant_value(df):
    
    constant_value_feature = []
    info = df.describe(include='all')
    for i in df.columns:
        if info[i]['std']==0:
            constant_value_feature.append(i)
    df.drop(constant_value_feature,axis=1,inplace=True)
    return df,constant_value_feature

datos , dropped_feature = constant_value(datos)
print("Caracteristicas eliminadas con 0 varianza (0 std. dev.): ",dropped_feature)
print("Dimensiones dataset: ",datos.shape)


#%%
#datos perdidos

# Se crea un diccionario con clave el nombre de las columnas,
# y values el porcentaje de datos perdidos
nan_count = {k:list(datos.isna().sum()*100/datos.shape[0])[i] for i,k in enumerate(datos.columns)}

# Se ordena el diccionario en orden descendente
nan_count = {k: v for k, v in sorted(nan_count.items(), key=lambda item: item[1],reverse=True)}

#Se muestran las 15 característica con porcentaje de datos perdidos más alto
sns.set_style(style="whitegrid")
plt.figure(figsize=(20,10))

# Gráfico de barras
plot = sns.barplot(x= list(nan_count.keys())[:15],y = list(nan_count.values())[:15],palette="hls")

# Se añade el porcentaje encima de cada columna
for p in plot.patches:
        plot.annotate('{:.1f}%'.format(p.get_height()), (p.get_x()+0.2, p.get_height()+1))


plot.set_yticklabels(map('{:.1f}%'.format, plot.yaxis.get_majorticklocs())) 
plt.show()

#%%
'''
    Esta función elimina las característica con más del 70%
    de datos perdidos, y elimina las filas que tienen
    valores NA de características que tienen  menos del 5% de
    datos perdidos
'''

# df: Dataframe con los datos del problema
# nan_feat: Lista de características que contienen menos del 5% NA
def remove_na(df,nan_feat):
    # Elimina características con más del 70% NA
    df = df.dropna(axis = 1, thresh=18000)
    
    # Elimina filas que contienen NA de la lista pasada, nan_feat
    df = df.dropna(subset=nan_feat)
    # Resetea los valores de los índices 
    df = df.reset_index(drop=True)
    return df

print("Tamaño del dataset previo eliminación de datos perdidos:",datos.shape)


# Lista de características que contienen menos del 5% NA
na_5 = [k for k,v in nan_count.items() if v < 5]

datos = remove_na(datos,na_5)
print("Dimension despues de eliminar filas y columnas:",datos.shape)
print("Numero de caracteristicas con menos del 5% de datos perdidos:",len(na_5))

# lista con las 7 características con mayor valor de datos perdidos
# creating a list of the top 7 features having highest number of missing values
na_70 = list(nan_count.keys())[:7]

# Total removed features
removed_features = na_70 + dropped_feature
print("Caracteristicas eliminadas:", removed_features)

#%%
y = datos['class']

X = datos.drop('class',axis=1)



# Lista de características con datos perdidos entre el 5 y el 70%
mis_col = [k for k,v in nan_count.items() if v >= 5 and v < 70]


median_imputer = SimpleImputer(missing_values=np.nan, strategy='median',copy=True)

# Dataframe con los valores imputados
median_df = median_imputer.fit_transform(X[mis_col])
X[mis_col] = median_df
    




#Lectura datos de test
X_test = pd.read_csv('datos/aps_failure_test_set.csv', skiprows=20, na_values=["na"])

nan_count_test = {k:list(X_test.isna().sum()*100/X_test.shape[0])[i] for i,k in enumerate(X_test.columns)}
nan_5_test = [k for k,v in nan_count_test.items() if v < 5]
X_test = X_test.dropna(subset = nan_5_test)
y_test = X_test['class']
y_test = y_test.replace(['neg', 'pos'], [0, 1])
X_test = X_test.drop('class', axis = 1)
#Se eliminan las columnas con mas del 70% de valores perdidos
X_test = X_test.drop(removed_features, axis = 1)
X_test[mis_col] = median_imputer.transform(X_test[mis_col])

  
print("Dimension del conjunto de test: ",X_test.shape)




#%%



#Preprocesado
over = SMOTE(sampling_strategy=0.3)
X_train, y_train = over.fit_resample(X, y)
under = RandomUnderSampler(sampling_strategy=0.5)
X_train, y_train = under.fit_resample(X_train, y_train)
print('Dimension despues de Smote y Undersampling: ', X_train.shape)
print('Numero de muestras de cada clase:')
print(y_train.value_counts())



scaler = StandardScaler()
X_train_prep = scaler.fit_transform(X_train)
X_test_prep = scaler.transform(X_test)



#%%
#Representa la matriz de confusión
def plot_confusion( y_test , y_hat ): 
    
    cf_matrix_test = confusion_matrix(y_test , y_hat)
        
    group_names = ["TN","FP","FN","TP"]
    group_counts = ["{}".format(value) for value in cf_matrix_test.flatten()]
    labels = [f"{v1}\n{v2}" for v1, v2 in zip(group_names,group_counts)]
    labels = np.asarray(labels).reshape(2,2)

    sns.heatmap(cf_matrix_test, annot=labels, fmt='', cmap='Blues')
    plt.show()
    
def model_results_pred( model , x_train , x_test , y_train , y_test ):
    
    y_train_hat = model.predict(x_train)
    y_test_hat = model.predict(x_test) 
    
    f1_macro = f1_score(y_test, y_test_hat,average='macro')
    
    print('\033[1m'+'Macro-F1 Score: ',f1_macro)
    
    # Pinta matriz de confusion
    plot_confusion(y_test,y_test_hat)
    
    return f1_macro

#%%

# modelos_lr = [LogisticRegression(C = 0.5, max_iter=1000, n_jobs=-1),
#               LogisticRegression(C = 1, max_iter=1000, n_jobs=-1),
#               LogisticRegression(C = 1.5, max_iter=1000, n_jobs=-1)
#               ]

# results = []
# for i in modelos_lr:
#     i.fit(X=X_train_prep, y=y_train) 
#     cv_scores = cross_val_score(
#         estimator = i, 
#         X = X_train_prep,
#         y = y_train,
#         scoring = 'f1_macro',
#         cv = 5,
#         n_jobs = -1)
#     results.append(cv_scores.mean())


m_lr = LogisticRegression(C = 1.5, max_iter=1000, n_jobs=-1)
m_lr.fit(X_train_prep, y_train)
F1_lr = model_results_pred(m_lr, X_train_prep, X_test_prep, y_train, y_test)
