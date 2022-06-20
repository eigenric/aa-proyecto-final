# -*- coding: utf-8 -*-

"""
Created on Sat Jun  4 09:11:30 2022

@author: Daniel Navarrete Martin
@author: Ricardo Ruiz Fernandez de Alba
"""
# Enlace ficheros de datos consigna UGR: https://consigna.ugr.es/f/0t2Ee6Pj5MG4JXdv/datos.rar

# Tratamiento de datos
# ==============================================================================
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE

import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.model_selection import cross_val_score
from sklearn.dummy import DummyClassifier

from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from imblearn.under_sampling import RandomUnderSampler
from lightgbm import LGBMClassifier

def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn


datos = pd.read_csv('datos/aps_failure_training_set.csv', skiprows=20, na_values=["na"])
datos['class'] = datos['class'].replace(['neg','pos'],[0,1])


# Distribución de la etiqueta class
# Neg -> 0
# Pos -> 1

#Se genera una gráfica con la distribución de la etiqueta 'class'
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

input('Pulse una tecla para continuar')


#datos perdidos

# Se crea un diccionario con clave el nombre de las columnas,
# y values el porcentaje de datos perdidos
nan_count = {k:list(datos.isna().sum()*100/datos.shape[0])[i] for i,k in enumerate(datos.columns)}

# Se ordena el diccionario en orden descendente
nan_count = {k: v for k, v in sorted(nan_count.items(), key=lambda item: item[1],reverse=True)}

#Se muestran las 15 características con porcentaje de datos perdidos más alto
sns.set_style(style="whitegrid")
plt.figure(figsize=(20,10))

# Gráfico de barras
plot = sns.barplot(x= list(nan_count.keys())[:15],y = list(nan_count.values())[:15],palette="hls")

# Se añade el porcentaje encima de cada columna
for p in plot.patches:
        plot.annotate('{:.1f}%'.format(p.get_height()), (p.get_x()+0.2, p.get_height()+1))


plot.set_yticklabels(map('{:.1f}%'.format, plot.yaxis.get_majorticklocs())) 
plt.show()


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
    df = df.dropna(axis=1, thresh=18000)
    
    # Elimina filas que contienen NA de la lista pasada, nan_feat
    df = df.dropna(subset=nan_feat)
    # Resetea los valores de los índices 
    df = df.reset_index(drop=True)
    return df

print('Eliminación de datos perdidos: ')
print("Tamaño del dataset previo eliminación de datos perdidos:",datos.shape)

# Lista de características que contienen menos del 5% NA
na_5 = [k for k,v in nan_count.items() if v < 5]
datos = remove_na(datos, na_5)

# Elimina las filas con más del 15% de valores perdidos
nan_thresh = int(0.85 * len(datos.columns))
datos = datos.dropna(thresh=nan_thresh)

print("Dimension despues de eliminar filas y columnas:",datos.shape)
print("Numero de caracteristicas con menos del 5% de datos perdidos:",len(na_5))

# lista con las 7 características con mayor valor de datos perdidos
na_70 = list(nan_count.keys())[:7]


removed_features = na_70 + dropped_feature
print("Caracteristicas eliminadas:", removed_features)
input('Pulse una tecla para continuar')



# Asignación datos de entrenamiento
y_train = datos['class']

X_train = datos.drop('class', axis=1)


# Lista de características con valores perdidos entre 5% y 15%
# Imputaremos los datos perdidos calculando su media
mis_col = [k for k,v in nan_count.items() if 5 < v < 70]
mean_imputer = SimpleImputer(missing_values=np.nan, strategy='mean',copy=True)

# Dataframe con los valores imputados
# Criterio: Valor medio más valor aleatorio uniforme
# en [-1.5sigma, 1.5sigma] con sigma cada desviación típica

mean_df = mean_imputer.fit_transform(X_train[mis_col])
stds = np.std(X_train, axis=0)[mis_col]
noise = np.random.uniform(-1.5*stds, 1.5*stds)
X_train[mis_col] = mean_df + noise

#Lectura datos de test
X_test = pd.read_csv('datos/aps_failure_test_set.csv', skiprows=20, na_values=["na"])

# Preparación de datos de test
nan_count_test = {k:list(X_test.isna().sum()*100/X_test.shape[0])[i] for i,k in enumerate(X_test.columns)}
nan_5_test = [k for k,v in nan_count_test.items() if v < 5]
X_test = X_test.dropna(subset = nan_5_test)
y_test = X_test['class']
y_test = y_test.replace(['neg', 'pos'], [0, 1])
X_test = X_test.drop('class', axis = 1)
#Se eliminan las columnas con mas del 70% de valores perdidos
X_test = X_test.drop(removed_features, axis = 1)

mean_df_test = mean_imputer.transform(X_test[mis_col])
noise_test = np.random.uniform(-1.5*stds, 1.5*stds)
X_test[mis_col] = mean_df_test + noise

  
print("Dimension del conjunto de test: ", X_test.shape)
input('Pulse una tecla para continuar')


#Preprocesado de los datos
print('Preprocesado de los datos:')
over = SMOTE(sampling_strategy=0.3)
X_train, y_train = over.fit_resample(X_train, y_train)

under = RandomUnderSampler(sampling_strategy=0.5)
X_train, y_train = under.fit_resample(X_train, y_train)
print('Dimension despues de Smote y Undersampling: ', X_train.shape)
print('Numero de muestras de cada clase:\n', y_train.value_counts())

scaler = StandardScaler()
X_train_prep = scaler.fit_transform(X_train)
X_test_prep = scaler.transform(X_test)
input('Pulse una tecla para continuar')



# MODELO LINEAL
# Regresion Logistica

print("Regresión Logística: ")

# Se crean varios modelos para estimar los hiperparámetros

# Los siguientes modelos se encuentran comentados para reducir el tiempo de ejecución.
# Se han tenido en cuenta en la memoria

modelos_lr = [# LogisticRegression(penalty = 'l1', C = 0.5, max_iter=1000,  solver='liblinear', random_state=30),
              # LogisticRegression(penalty = 'l1', C = 1, max_iter=1000, solver = 'liblinear', random_state=30),
              LogisticRegression(penalty = 'l1', C = 1.5, max_iter=1000, solver = 'liblinear', random_state=30),
              # LogisticRegression(penalty = 'l1', C = 0.5, tol = 0.001, max_iter=1000, solver='liblinear', random_state=30),
              # LogisticRegression(penalty = 'l1', C = 1, tol = 0.001, max_iter=1000, solver='liblinear', random_state=30),
              LogisticRegression(penalty = 'l1', C = 1.5, tol = 0.001, max_iter=1000,  solver='liblinear', random_state=30)
              ]

resultsFinal = dict()

import time
start_time = time.time()

results = []
cont = 1
best_lr = None
best_result = 0
# Se realiza Validación Cruzada con los distintos modelos

for i in modelos_lr:    
    i.fit(X=X_train_prep, y=y_train) 
    cv_scores = cross_val_score(
          estimator = i, 
          X = X_train_prep,
          y = y_train,
          scoring = 'f1_macro',
          cv = 5,
          n_jobs = -1)
    
    results.append(cv_scores.mean())
    result = cv_scores.mean()
    if best_result < result:
        best_lr = i
        best_result = result
    results.append(result)
    print('Resultado ' + str(cont) + 'º modelo:', result)
    cont += 1

print("--- %s seconds ---" % (time.time() - start_time))

# Se guarda el mejor resultado obtenido por LR
resultsFinal["RL"] = best_result

m_lr = LogisticRegression(penalty = 'l1', C = 1.5, max_iter=1000, solver = 'liblinear', random_state=30),
input('Pulse una tecla para continuar')



def VC_k_fold(X, y, model, params, cv=5):
    """Realiza validación cruzada 5-fold. Devuelve la puntuación media"""

    model.set_params(**params)
                       
    cv = RepeatedStratifiedKFold(n_splits=cv, random_state=1)
    scores = cross_val_score(model, X, y, scoring='f1_macro',
                             cv=cv, n_jobs=-1)
                            
    return np.mean(scores)


# MODELOS NO LINEALES

# Random Forest
print('Random Forest:')
def ajuste_rf_estimator(params, step_nestimators=100, step_max_depth=10):
    """Siguientes hiperparámetros del espacio de búsqueda"""

    nestimators = params["n_estimators"]
    new_nestimators = nestimators + step_nestimators

    max_depth = params["max_depth"]
    new_max_depth = max_depth + step_max_depth

    params.update({"n_estimators": new_nestimators, 
                   "max_depth": new_max_depth})

    return params


def tuning_rf(X, y, model, params, step_nestimators=100, step_max_depth=10,
              cv=5, n_repeat=3):
    """Ajusta los parametros en el espacio de búsqueda"""
    
    results_rf = pd.DataFrame(params, index=[0])
    
    best_params = params.copy()
    best_score = VC_k_fold(X, y, model, params, cv=cv)
    results_rf["score"] = best_score
    
    for i in range(n_repeat-1):
        params = ajuste_rf_estimator(params, 
                                     step_nestimators=step_nestimators,
                                     step_max_depth=step_max_depth)
        
        score = VC_k_fold(X, y, model, params, cv=cv)
        
        params_score = {**params, "score": score}
        results_rf = results_rf.append(params_score, 
                                       ignore_index=True)

        if score >= best_score:
            best_params, best_score = params.copy(), score
    
    return best_params, best_score, results_rf

params = {'n_estimators': 5,
          'max_depth': 25}

start_time = time.time()

m_rf = RandomForestClassifier(n_jobs=-1, verbose=0)

# Obtención de los mejores hiperparámetros
best_params_rf, best_score_rf, results_rf = tuning_rf(X_train_prep,
                                                      y_train,
                                                      m_rf,
                                                      params,
                                                      step_nestimators=5,
                                                      step_max_depth=20,
                                                      n_repeat=3,
                                                      cv=5)

print("--- %s seconds ---" % (time.time() - start_time))

resultsFinal["RF"] = best_score_rf
results_rf.sort_values(by=["score", "n_estimators"], 
                       ascending=False, 
                       inplace=True)
print(results_rf)
input('Pulse una tecla para continuar')



# Gradient Boosting 
print('Gradient Boosting:')
def ajuste_gb_estimator(params, step_nestimators=10, 
                        step_max_depth=1, step_lr=0.1):
    """Siguientes hiperparámetros del espacio de búsqueda"""

    params["n_estimators"] +=  step_nestimators
    params["max_depth"] += step_max_depth
    params["learning_rate"] += step_lr
    
    return params


def tuning_gb(X, y, model, params, step_nestimators=100, 
              step_max_depth=1, step_lr=0.1,
              cv=5, n_repeat=3):
    """Ajusta los parametros en el espacio de búsqueda"""
    
    results_gb = pd.DataFrame(params, index=[0])
    
    best_params = params.copy()
    print(f"0-ésima GB con {best_params}")
    start_time = time.time()
    best_score = VC_k_fold(X, y, model, params, cv=cv)
    print("--- %s seconds ---" % (time.time() - start_time))
    print(f"Score => {best_score}")
    results_gb["score"] = best_score
    
    for i in range(n_repeat-1):
        params = ajuste_gb_estimator(params, 
                                     step_nestimators=step_nestimators,
                                     step_max_depth=step_max_depth,
                                     step_lr=step_lr)

        print(f"{i+1}-ésima GB con {params}")             
        start_time = time.time()
        score = VC_k_fold(X, y, model, params, cv=cv)
        print("--- %s seconds ---" % (time.time() - start_time))
        print(f"Score => {score}")
        
        params_score = {**params, "score": score}
        results_gb = results_gb.append(params_score, 
                                       ignore_index=True)

        if score >= best_score:
            best_params, best_score = params.copy(), score
    
    return best_params, best_score, results_gb

# Se prueban los parametros [200, 300, 400], [2, 3, 4]
params = {'n_estimators': 200,
          'max_depth': 2,
          'learning_rate': 0.1}

import time
start_time = time.time()

m_gb = LGBMClassifier(n_jobs=-1,
                      random_state=42)

# Obtención de los mejores hiperparámetros
best_params_gb, best_score_gb, results_gb = tuning_gb(X_train_prep,
                                                      y_train,
                                                      m_gb,
                                                      params,
                                                      step_nestimators=100,
                                                      step_max_depth=1,
                                                      step_lr=0,
                                                      n_repeat=3,
                                                      cv=5)


resultsFinal["GB"] = best_score_gb
results_gb.sort_values(by=["score", "n_estimators"], 
                       ascending=False, 
                       inplace=True)
print(results_gb)
print("--- %s seconds ---" % (time.time() - start_time))
input('Pulse una tecla para continuar')

# Support Vector Machine. 

def ajuste_lr_alpha(params, step_size=0.1):
    """Siguientes hiperparámetros del espacio de búsqueda"""

    alpha, alpha_or = params["alpha"], params["alpha"]
    new_alpha = alpha +  step_size
    new_alpha = alpha_or if new_alpha <= 0.0 else new_alpha

    new_lr_alpha = {"alpha": new_alpha}
    params.update(new_lr_alpha)

    return params

def tuning(X, y, model, params, step_size=0.1, cv=5, n_repeat=3):
    """Ajusta los parametros en el espacio de búsqueda"""
    
    results_svm = pd.DataFrame(params, index=[0])

    best_params = params.copy()
    best_score = VC_k_fold(X, y, model, params, cv=cv)
    results_svm["score"] = best_score
    
    
    for i in range(n_repeat-1):
        params = ajuste_lr_alpha(params, step_size)
        score = VC_k_fold(X, y, model, params, cv=cv)
        params_score = {**params, "score": score}
        results_svm = results_svm.append(params_score, 
                                         ignore_index=True)
        
        if score >= best_score:
            best_params, best_score = params.copy(), score
    
    return best_params, best_score, results_svm

print("Support Vector Machine (SVM): ")

# Parámetros iniciales del espacio de búsqueda
params = {"penalty": "l2", 
          "alpha": 2.5e-3, 
          "tol" : 0.1,
          "max_iter": 1000}

params_or = params.copy()

import time
start_time = time.time()
# Obtención de los mejores hiperparámetros paso a paso
m_svm = SGDClassifier(loss="hinge", n_jobs=-1, random_state=0)


ss = 5e-5
nr = 3
best_params_svm, best_score_svm, results_svm = tuning(X_train_prep, y_train,
                                                      m_svm,
                                                      params,
                                                      step_size=ss,
                                                      n_repeat=nr,
                                                      cv=5)

print("--- %s seconds ---" % (time.time() - start_time))
resultsFinal["SVM"] = best_score_svm
results_svm.sort_values(by=["score", "alpha"], 
                        ascending=False, 
                        inplace=True)
print(results_svm)
input('Pulse una tecla para continuar')


# Elección del mejor modelo
#Representa la matriz de confusión
def plot_confusion(y_test, y_hat):
    """
    Matriz de Confusion según las etiquetas 
    verdaderas y predichas.
    """
    
    fig = plt.figure()
    cf_matrix_test = confusion_matrix(y_test , y_hat)
        
    group_names = ["TN","FP","FN","TP"]
    group_counts = ["{}".format(value) for value in cf_matrix_test.flatten()]
    labels = [f"{v1}\n{v2}" for v1, v2 in zip(group_names,group_counts)]
    labels = np.asarray(labels).reshape(2,2)

    sns.heatmap(cf_matrix_test, annot=labels, fmt='', cmap='Blues')
    plt.show()
    
def model_results_pred(model, x_train , x_test , y_train , y_test ):
    """
    Esta funcion predice la etiqueta de clase y devuelve
    la puntuación Macro-F1
    """
    
    y_test_hat = model.predict(x_test) 
    
    f1_macro = f1_score(y_test, y_test_hat, average='macro')
    
    print('\033[1m'+'Macro-F1 Score: ',f1_macro)
    
    # Pinta la matriz de confusión
    print("\tTest Confusion Matrix")
    plot_confusion(y_test,y_test_hat)
    
    return f1_macro


# Baseline para comparar el resto de modelos

baseline = DummyClassifier(strategy='constant', constant=0)
baseline.fit(X_train_prep, y_train)

cv_scores = cross_val_score(
      estimator = baseline, 
      X = X_train_prep,
      y = y_train,
      scoring = 'f1_macro',
      cv = 5,
      n_jobs = -1)
resultsFinal = dict()
resultsFinal['Baseline'] = cv_scores.mean()

print("Baseline: ")
F1_Base = model_results_pred(baseline, 
                             X_train_prep, X_test_prep, 
                             y_train, y_test)

resultsFinal = sorted(resultsFinal.items(), key=lambda m: m[1],
                      reverse=True)

# Evaluacion del mejor modelo en test

# Curva de aprendizaje
m_gb.set_params(**best_params_gb)
evalset = [(X_train_prep, y_train), (X_test_prep, y_test)]
m_gb.fit(X_train_prep, y_train, eval_set=evalset, verbose=0)
yhat = m_gb.predict(X_test)
score = f1_score(y_test, yhat)
result_LC = m_gb.evals_result_
fig = plt.figure()
plt.plot(result_LC['training']['binary_logloss'], label='train')
plt.plot(result_LC['valid_1']['binary_logloss'], label='test')
plt.xlabel('Nº de estimadores')
plt.ylabel('Función de pérdida log-loss')
plt.legend()
# show the plot
plt.show()


E_test = model_results_pred(m_gb, X_train_prep, X_test_prep, y_train, y_test)



