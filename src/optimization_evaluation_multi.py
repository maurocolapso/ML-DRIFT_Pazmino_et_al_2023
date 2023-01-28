from locale import normalize
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import seaborn as sn
import numpy as np


def model_optimization_SVC_multiclass(X, y):

    # Define a Standard Scaler to normalize inputs
    scaler = StandardScaler()

# set the tolerance to a large value to make the example faster
    rf = SVC(class_weight='balanced')

    pipe = Pipeline(steps=[("scaler", scaler), ("rf", rf)])

    # Parameters of pipelines can be set using ‘__’ separated parameter names:
    param_grid = {'rf__kernel': ['linear','rbf', 'poly', 'sigmoid'],
                  'rf__C': [0.1, 1, 10, 100],
                  'rf__gamma': [1, 0.1, 0.01, 0.001]}

    search = GridSearchCV(pipe, param_grid, n_jobs=2,scoring='accuracy',cv=5)
    search.fit(X, y.ravel())
    print("Best parameter (CV score=%0.3f):" % search.best_score_)
    print(search.best_params_)
    return pipe, search

def model_evaluation_SVC_multiclass(X_tr, y_tr, X_ts, y_ts):
    scaler = StandardScaler()

    # set the tolerance to a large value to make the example faster
    rf = SVC(C=0.1, gamma= 1, kernel= 'linear', class_weight='balanced')

    pipe = Pipeline(steps=[("scaler", scaler), ("rf", rf)])

    pipe.fit(X_tr, y_tr.ravel())

    #variable_importance = pipe.named_steps["rf"].feature_importances_

    eval_score = pipe.score(X_ts, y_ts)
    
    y_pred = pipe.predict(X_ts)

    return eval_score, y_pred

def confusion_matrix_multiclass(y_true, y_pred):
    local_report= classification_report(y_true, y_pred)
    print(local_report)

    sn.set_context("talk", font_scale=1.2)

    fig, ax = plt.subplots(figsize=(6,5))
    disp = ConfusionMatrixDisplay.from_predictions(y_true, y_pred, normalize='true', cmap='PuBu', ax=ax)
    #plt.yticks(rotation=45)
    #plt.xticks([0.5,1.5],label_names, ha = 'center')
    #plt.yticks([0.5,1.5],label_names)

def confusion_matrix_nested(cm_nested, labels):

    cm_nested_normalized = cm_nested.astype('float')/cm_nested.sum(axis=1)[:, np.newaxis]
    sn.set_context("talk", font_scale=1.5)

    fig, ax = plt.subplots(figsize=(6,5))
    disp = ConfusionMatrixDisplay(cm_nested_normalized, display_labels=labels)
    disp.plot(cmap = "PuBu", ax=ax)
    #plt.xticks(rotation=45)
    #plt.xticks([0.5,1.5],label_names, ha = 'center')
    #plt.yticks([0.5,1.5],label_names)