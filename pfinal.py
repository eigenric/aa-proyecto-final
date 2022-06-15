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
# import joblib
# from sklearn.dummy import DummyClassifier
# from sklearn.linear_model import LogisticRegression
# from sklearn.model_selection import RandomizedSearchCV
# from sklearn.model_selection import GridSearchCV
# from sklearn.metrics import roc_curve,auc
# from sklearn.metrics import confusion_matrix
# from sklearn.naive_bayes import GaussianNB
# from scipy.stats import uniform,randint
# from tqdm import tqdm
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.ensemble import RandomForestClassifier
# from xgboost import XGBClassifier
# from sklearn.preprocessing import MinMaxScaler
# from lightgbm import LGBMClassifier
# from sklearn.metrics import f1_score
# from sklearn.metrics import precision_recall_curve
from imblearn.under_sampling import RandomUnderSampler
# from imblearn.pipeline import Pipeline
# from sklearn.linear_model import SGDClassifier
# from sklearn.ensemble import AdaBoostClassifier
# from prettytable import PrettyTable
# import pickle


import warnings
warnings.filterwarnings("ignore")

#Ejercicio de Regresion

#Lectura de datos


datos = pd.read_csv('datos/aps_failure_training_set.csv', skiprows=20, na_values=["na"])

# y = datos['class']

# y[y == 'neg'] = 0
# y[y == 'pos'] = 1

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
    """
    This function returns a list of columns
    that have std. deviation of 0
    meaning, all values are constant
    """
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
    
    # Elimina files que contienen NA de la lista pasada, nan_feat
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

def imputation(df,mis_col):
    """
    This function imputes Missing values 
    using Median on given features, and
    Model Based Imputation on the rest
    
    """
    
    # Using sklearn's SimpleImputer
    median_imputer = SimpleImputer(missing_values=np.NaN , strategy='median',copy=True)

    # Creating a new dataframe of imputed values
    median_df = median_imputer.fit_transform(df[mis_col])
    df1 = df.copy()
    df1[mis_col] = median_df

    # Performing Model-Based Imputation
    mice_imputer = IterativeImputer(estimator=Ridge(random_state=0),
                                    random_state=0)
    imputed_df = pd.DataFrame(data = mice_imputer.fit_transform(df1) , columns= df1.columns )

    return imputed_df , median_imputer , mice_imputer


# List of feature names that have missing values between 5% to 15%.
# We will impute the missing values in features with their median
median_imputed_features = [k for k,v in nan_count.items() if v >= 5 and v < 15]

imputed_x_train , MEDIAN_imputer , MICE_imputer = imputation( X , median_imputed_features )
print("Number of features whose missing values are imputed with median are:\n",len(median_imputed_features))








#%%

dat_arr = np.array(y)
labels = np.array([0 if i=='neg' else 1 for i in dat_arr])
X = imputed_x_train
X_train, X_test, y_train, y_test = train_test_split(
    X,
    labels,
    train_size = 0.8,
    random_state= 1, 
    shuffle = True)

#Preprocesado
over = SMOTE(sampling_strategy=0.3)
under = RandomUnderSampler(sampling_strategy=0.5)
preprocessor = ColumnTransformer(
    [('over', over),
     ('under', under)],
    remainder = 'passthrough')

X_train_prep, y_train_prep = preprocessor.fit(X_train, y_train)
pipeline = Pipeline(steps = 
                    [('over', over),
                     ('under', under)],
                    remainder = 'passthrough')

X_train_prep, y_train_prep = pipeline.fit(X_train, y_train)
print(X_train_prep.shape, y_train_prep.value_counts())
# scaler = StandardScaler()
# X_train_prep = scaler.fit_transform(X_train_prep)


# num_cols = X_train.select_dtypes(include=['int', 'float']).columns.to_list()

# preprocessor = ColumnTransformer(
#     [('numeric', StandardScaler(), num_cols)],
#     remainder = 'passthrough')

# x_train_prep = preprocessor.fit_transform(x_train)
# x_test_prep = preprocessor.transform(x_test)

# m_lr = LogisticRegression(class_weight='balanced', max_iter=1000)

# m_lr = m_lr.fit(X=x_train_prep, y=y_train)

# cv_scores = cross_validate(
#     estimator = m_lr,
#     X = x_train_prep,
#     y = y_train,
#     n_jobs = -1,
#     scoring = ('f1_macro'),
#     cv = 5)
# lr_scores = pd.DataFrame(cv_scores)
# lr_mean_scores = np.array(lr_scores)
