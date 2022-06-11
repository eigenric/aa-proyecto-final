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
# from sklearn.model_selection import train_test_split
# from collections import Counter
# from sklearn.impute import SimpleImputer
# from sklearn.feature_selection import RFE
import matplotlib.pyplot as plt
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.experimental import enable_iterative_imputer
# from imblearn.over_sampling import SMOTE
# from sklearn.impute import IterativeImputer
# from sklearn.linear_model import Ridge
import seaborn as sns
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
# from imblearn.under_sampling import RandomUnderSampler
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

y = datos['class']

y[y == 'neg'] = 0
y[y == 'pos'] = 1

# Plotting the distribution of class label
sns.barplot(y.unique(),y.value_counts())
plt.title('Class Label Distribution')
plt.xlabel('Class Label')
plt.ylabel('Count')
plt.show()

print('The number of positive class points is: ',y.value_counts()[1])
print('The number of negative class points is: ',y.value_counts()[0])




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
print("The features that are dropped due to having a constant value (0 std. dev.) are: ",dropped_feature)
print("Shape of our feature set: ",datos.shape)

X = datos.drop('class', axis = 1)

#%%
arr = np.array(datos)

aux = arr[:,0]
n_neg = len((np.where(aux=='neg'))[0])

print(n_neg, len(arr[:,0]) - n_neg)

#Dimensiones del dataset
print("\n-----------------------------------")
print('Dimensiones del dataset:', datos.shape)


#Valores nulos
print("\n-----------------------------------")
print('Datos ausentes por variables:')
print(datos.isna().sum().sort_values())

