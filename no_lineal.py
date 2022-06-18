def VC_k_fold(X, y, model, params, cv=5):
    """Realiza validación cruzada 5-fold"""

    model.set_params(**params)
                       
    cv = RepeatedStratifiedKFold(n_splits=5, random_state=1)
    scores = cross_val_score(model, X, y, scoring='f1_score',
                             cv=cv, n_jobs=-1)
                            
    return np.mean(scores)


# Ajuste SVM

def ajuste_lr_alpha(params, step_size):
    """Siguientes hiperparámetros"""

    lr, alpha = params["eta0"], params["alpha"]
    new_lr = lr + np.random.randn() * step_size
    new_lr = 1e-8 if new_lr <= 0.0 else lr
    new_alpha = alpha + np.random.randn() * step_size
    new_alpha = 0.0 if new_alpha < 0.0 else alpha

    new_lr_alpha = {"eta0": new_lr, "alpha": new_alpha} 
    params.update(new_lr_alpha)

    return params


def tuning(X, y, model, params, step_size, cv=5, n_repeat=3):
    """Ajusta los parametros en el espacio de búsqueda"""

    params = [np.random.rand(), np.random.rand()]
    best_score = VC_k_fold(X, y, model, params)

    for i in range(n_repeat):
        params = ajuste_lr_alpha(solution, step_size)
        score = VC_k_fold(X, y, model, params, cv=cv)

        if score >= best_score:
            best_params, best_score = params, score
    
    return best_params, best_score	

def plot_confusion(y_test, y_pred):
    """Genera Matriz de Confusión según las etiquetas predichas y verdaderas"""    

    # Mapa de Calor
    cf_matrix_test = confusion_matrix(y_test, y_pred)
        
    group_names = ["TN","FP","FN","TP"]
    group_counts = ["{}".format(value) for value in cf_matrix_test.flatten()]
    labels = [f"{v1}\n{v2}" for v1, v2 in zip(group_names,group_counts)]
    labels = np.asarray(labels).reshape(2,2)

    sns.heatmap(cf_matrix_test, annot=labels, fmt='', cmap='Blues')
    plt.show()


def model_results_pred(model, x_train, x_test, y_train, y_test):
    """Predice la etiqueta de los datos y devuelve su puntuación macro-F1"""

    # Predicción de las etiquetas de clase
    y_train_pred = model.predict(x_train)
    y_test_pred = model.predict(x_test) 
    
    f1_macro = f1_score(y_test, y_test_pred, average='macro')
    
    print(f"Macro-F1 Score: {f1_macro}")
    
    print("\tTest Confusion Matrix")
    plot_confusion(y_test, y_test_pred)
    
    return f1_macro

#%%
baseline = DummyClassifier(strategy='constant', constant=0)
baseline.fit(X_train_prep, y_train_prep)

F1_Base = model_results_pred(baseline, 
                             X_train_prep, X_test_prep, 
                             y_train_prep, y_test_prep)

# Parámetros iniciales del espacio de búsqueda
params = {"penalty": "l1", 
          "alpha": 0.0, 
          "eta0": 1e-8}

# Obtención de los mejores hiperparámetros
best_params_svm, best_score_svm = tuning(X_train_prep, y_train_prep,
                                         SGDClassifier(loss="hinge", 
                                                       n_jobs=-1,
                                                       random_state=0),
                                         params,
                                         cv=2,
                                         verbose=1)

print(f"Mejores Parámetros:  {best_params_svm} with score of: {best_score_svm}")
