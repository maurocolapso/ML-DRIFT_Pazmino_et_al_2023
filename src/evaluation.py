from pyexpat import features
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sn
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from transformers import MultipleScatterCorrection, SavitzkyGolay
from transformers import StandardNormalVariate

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import RocCurveDisplay
from sklearn.metrics import auc

from matplotlib.ticker import FormatStrFormatter

def wavenumberlist(X):
        X_wavenumbers = X
        dfc = X_wavenumbers.T
        wv2 = (list(dfc.index.values))
        wvn2 = [int(x) for x in wv2]
        return wvn2


def variable_contribution_top10_plot(features_importance):
    """"Plot variable importance/coefficients vs wavenumbers
    Parameters
    ----------
    features_importance: dataframe

    Return
    ------
     variable_importance_sort: dataframe
    """
    sn.set_theme(style="ticks")
    sn.color_palette("tab10")
    sn.set_context("notebook", font_scale=1.4)

    fig,ax = plt.subplots(figsize=(10,5))
    ax.plot(features_importance.iloc[:,0], features_importance.iloc[:,1], color="darkgreen")
    ax.set_xlim(1800,600)
    ax.set_xlabel('Wavelenght (cm$^{-1}$)')
    ax.set_ylabel('Random Forest feature importance')

    variable_importance_sort = features_importance.sort_values(by=["Coefficients"], ascending=False).head(10)

    for row in variable_importance_sort.itertuples():
        ax.annotate(row[1],
                    xy=(row[1], row[2]), xycoords='data',
                    xytext=(row[1], row[2]), textcoords='data',
                    arrowprops=dict(arrowstyle="->",
                            connectionstyle="arc3"),
                    fontsize=12)
    
    return variable_importance_sort


def model_evaluation(X_tr, y_tr, X_ts, y_ts):
    scaler = StandardScaler()
    derivative = SavitzkyGolay(filter_win=11, deriv_order=0)

    # set the tolerance to a large value to make the example faster
    rf = LogisticRegression(C = 1000, class_weight = {1: 0.4, 0: 0.6}, penalty = 'l2', solver = 'liblinear')

    pipe = Pipeline(steps=[("scaler", scaler), ("derivative", derivative), ("rf", rf)])

    pipe.fit(X_tr, y_tr.ravel())

    variable_importance = pipe.named_steps["rf"].coef_

    eval_score = pipe.score(X_ts, y_ts)
    
    y_pred = pipe.predict(X_ts)

    return eval_score, y_pred, variable_importance

def model_evaluation_species(X_tr, y_tr, X_ts, y_ts):
    scaler = StandardScaler()
    derivative = SavitzkyGolay(filter_win=9, deriv_order=2)

    # set the tolerance to a large value to make the example faster
    rf = RandomForestClassifier(bootstrap=True, max_depth=110, max_features= 3, min_samples_leaf=4, min_samples_split=12, n_estimators=200)


    pipe = Pipeline(steps=[("scaler", scaler), ("derivative", derivative), ("rf", rf)])

    pipe.fit(X_tr, y_tr.ravel())

    variable_importance = pipe.named_steps["rf"].feature_importances_

    eval_score = pipe.score(X_ts, y_ts)
    
    y_pred = pipe.predict(X_ts)

    return eval_score, y_pred, variable_importance


def model_evaluation_status(X_tr, y_tr, X_ts, y_ts):
    scaler = StandardScaler()
    derivative = SavitzkyGolay(filter_win=9, deriv_order=1)

    # set the tolerance to a large value to make the example faster
    rf = SVC(C=10, gamma= 1, kernel= 'sigmoid', class_weight='balanced')

    pipe = Pipeline(steps=[("scaler", scaler), ("derivative", derivative), ("rf", rf)])

    pipe.fit(X_tr, y_tr.ravel())

    #variable_importance = pipe.named_steps["rf"].feature_importances_

    eval_score = pipe.score(X_ts, y_ts)
    
    y_pred = pipe.predict(X_ts)

    return eval_score, y_pred



# nested cross validation 
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score



def nested_crossvalidation(X, y):
    """"Perform nested cross validation on a classifier
    """

    scaler = StandardScaler()
    model = LogisticRegression(class_weight={1: 0.6, 0: 0.4},solver='liblinear', random_state=1,max_iter=10000)


    pipe = Pipeline(steps=[("scaler", scaler), ("model", model)])

    # create confusion matrix list to save each of external cv layer
    cm_nested = []

    # configure nested cross-validation layers
    cv_outer = KFold(n_splits=10, shuffle=True, random_state=123)
    cv_inner = KFold(n_splits=3, shuffle=True, random_state=123)

    # enumerate splits and create AUC plot
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)
    outer_results = list()

    
    sn.set_style("ticks")
    sn.set_context("talk", font_scale=1.5)
    
    fig, ax = plt.subplots(figsize=(6, 5),constrained_layout=True)

    ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))

    for train_ix, test_ix in cv_outer.split(X):
    # split data
        X_train, X_test = X.iloc[train_ix, :], X.iloc[test_ix, :]
        y_train, y_test = y[train_ix], y[test_ix]

        # define search space
        param_grid = {'model__penalty': ['l1', 'l2'], 'model__C': [100, 1000]}
        # define search
        search = GridSearchCV(pipe, param_grid, scoring='roc_auc', cv=cv_inner, refit=True)
        # execute search
        result = search.fit(X_train, y_train)
        
        # get the best performing model fit on the whole training set
        best_model = result.best_estimator_

        # create pipeline with the best model
        best_pipe = Pipeline(steps=[("scaler", scaler), ("best_model", best_model)])
        best_pipe.fit(X_train, y_train)
        # evaluate model on the hold out dataset

        yhat = best_pipe.predict(X_test)
    
        viz = RocCurveDisplay.from_estimator(
            best_pipe,
            X_test,
            y_test,
            alpha=0.4,
            lw=1,
            label="_",
            ax=ax)

        interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
        aucs.append(viz.roc_auc)

        # evaluate the model
        acc = accuracy_score(y_test, yhat)
        cm = confusion_matrix(y_test, yhat)

        # store the result
        outer_results.append(acc)
        cm_nested.append(cm)


        # report progress
        print('>acc=%.3f, est=%.3f, cfg=%s' % (acc, result.best_score_, result.best_params_))

    # summarize the estimated performance of the model
    print('Accuracy: %.3f (%.3f)' % (np.mean(outer_results), np.std(outer_results)))

    ax.plot([0, 1], [0, 1], linestyle="--", lw=2, color="r", alpha=0.8)
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    ax.plot(
        mean_fpr,
        mean_tpr,
        color="b",
        label=r"Mean ROC (AUC = %0.2f $\pm$ %0.2f)" % (mean_auc, std_auc),
        lw=2.5,
        alpha=1)
    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    ax.fill_between(
        mean_fpr,
        tprs_lower,
        tprs_upper,
        color="grey",
        alpha=0.2)
    ax.set(
        xlim=[-0.05, 1.05],
        ylim=[-0.05, 1.05],
        ylabel="True positive rate",
        xlabel="False positive rate")
    #handles, lables = ax.get_legend_handles_labels()
    #line = Line2D([0], [0], label='manual line', color='k')
    #handles.extend([line])

    ax.legend(fontsize=15)
    #ax.get_legend().remove()
    return cm_nested

def nested_crossvalidation_species(X, y):
    """"Perform nested cross validation on a classifier
    """

    scaler = MultipleScatterCorrection()
    derivative = SavitzkyGolay()
    model = RandomForestClassifier()


    pipe = Pipeline(steps=[("scaler", scaler), ("derivative", derivative), ("model", model)])

    # configure nested cross-validation layers
    cv_outer = KFold(n_splits=10, shuffle=True, random_state=123)
    cv_inner = KFold(n_splits=3, shuffle=True, random_state=123)

    # create confusion matrix list to save each of external cv layer
    cm_nested = []

    # enumerate splits and create AUC plot
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)
    outer_results = list()

    sn.set_style("ticks")
    sn.set_context("talk", font_scale=1.5)
    
    fig, ax = plt.subplots(figsize=(6, 5),constrained_layout=True)
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))

    for train_ix, test_ix in cv_outer.split(X):
    # split data
        X_train, X_test = X.iloc[train_ix, :], X.iloc[test_ix, :]
        y_train, y_test = y[train_ix], y[test_ix]

        # define search space
        param_grid = {'model__bootstrap': [True],
                  'model__max_depth': [80, 90, 100, 110],
                  'model__max_features': [2, 3],
                  'model__min_samples_leaf': [3, 4, 5],
                  'model__min_samples_split': [8, 10, 12],
                  'model__max_depth': [80, 90, 100, 110],
                  'model__n_estimators': [100, 200, 300, 1000]}
        # define search
        search = GridSearchCV(pipe, param_grid, scoring='roc_auc', cv=cv_inner, refit=True)
        # execute search
        result = search.fit(X_train, y_train)
        
        # get the best performing model fit on the whole training set
        best_model = result.best_estimator_

        # create pipeline with the best model
        best_pipe = Pipeline(steps=[("scaler", scaler), ("best_model", best_model)])
        best_pipe.fit(X_train, y_train)
        # evaluate model on the hold out dataset

        yhat = best_pipe.predict(X_test)
    
        viz = RocCurveDisplay.from_estimator(
            best_pipe,
            X_test,
            y_test,
            alpha=0.4,
            lw=1,
            label="_",
            ax=ax)

        interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
        aucs.append(viz.roc_auc)

        # evaluate the model
        acc = accuracy_score(y_test, yhat)
        cm = confusion_matrix(y_test, yhat)

        # store the result
        outer_results.append(acc)
        cm_nested.append(cm)


        # report progress
        print('>acc=%.3f, est=%.3f, cfg=%s' % (acc, result.best_score_, result.best_params_))

    # summarize the estimated performance of the model
    print('Accuracy: %.3f (%.3f)' % (np.mean(outer_results), np.std(outer_results)))

    ax.plot([0, 1], [0, 1], linestyle="--", lw=2, color="r", alpha=0.8)
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    ax.plot(
        mean_fpr,
        mean_tpr,
        color="b",
        label=r"Mean ROC (AUC = %0.2f $\pm$ %0.2f)" % (mean_auc, std_auc),
        lw=2.5,
        alpha=1)
    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    ax.fill_between(
        mean_fpr,
        tprs_lower,
        tprs_upper,
        color="grey",
        alpha=0.2)
    ax.set(
        xlim=[-0.05, 1.05],
        ylim=[-0.05, 1.05],
        ylabel="True positive rate",
        xlabel="False positive rate")
    #handles, lables = ax.get_legend_handles_labels()
    #line = Line2D([0], [0], label='manual line', color='k')
    #handles.extend([line])

    ax.legend(fontsize=15)
    #ax.get_legend().remove()
    return cm_nested


def nested_crossvalidation_status(X, y):
    """"Perform nested cross validation on a classifier
    """

    scaler = StandardScaler()
    derivative = SavitzkyGolay(filter_win=9, deriv_order=1)
    model = SVC()


    pipe = Pipeline(steps=[("scaler", scaler), ("derivative", derivative), ("model", model)])

    # configure nested cross-validation layers
    cv_outer = KFold(n_splits=10, shuffle=True, random_state=123)
    cv_inner = KFold(n_splits=3, shuffle=True, random_state=123)

    # create confusion matrix list to save each of external cv layer
    cm_nested = []

    # enumerate splits and create AUC plot
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)
    outer_results = list()

    sn.set_style("ticks")
    sn.set_context("talk", font_scale=1.5)
    
    fig, ax = plt.subplots(figsize=(6, 5),constrained_layout=True)
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))

    for train_ix, test_ix in cv_outer.split(X):
    # split data
        X_train, X_test = X.iloc[train_ix, :], X.iloc[test_ix, :]
        y_train, y_test = y[train_ix], y[test_ix]

        # define search space
        param_grid = {'model__kernel': ['rbf', 'poly', 'sigmoid'],
                  'model__C': [0.1, 1, 10, 100],
                  'model__gamma': [1, 0.1, 0.01, 0.001]}
        # define search
        search = GridSearchCV(pipe, param_grid, scoring='roc_auc', cv=cv_inner, refit=True)
        # execute search
        result = search.fit(X_train, y_train)
        
        # get the best performing model fit on the whole training set
        best_model = result.best_estimator_

        # create pipeline with the best model
        best_pipe = Pipeline(steps=[("scaler", scaler), ("best_model", best_model)])
        best_pipe.fit(X_train, y_train)
        # evaluate model on the hold out dataset

        yhat = best_pipe.predict(X_test)
    
        viz = RocCurveDisplay.from_estimator(
            best_pipe,
            X_test,
            y_test,
            alpha=0.4,
            lw=1,
            label="_",
            ax=ax)

        interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
        aucs.append(viz.roc_auc)

        # evaluate the model
        acc = accuracy_score(y_test, yhat)
        cm = confusion_matrix(y_test, yhat)

        # store the result
        outer_results.append(acc)
        cm_nested.append(cm)


        # report progress
        print('>acc=%.3f, est=%.3f, cfg=%s' % (acc, result.best_score_, result.best_params_))

    # summarize the estimated performance of the model
    print('Accuracy: %.3f (%.3f)' % (np.mean(outer_results), np.std(outer_results)))

    ax.plot([0, 1], [0, 1], linestyle="--", lw=2, color="r", alpha=0.8)
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    ax.plot(
        mean_fpr,
        mean_tpr,
        color="b",
        label=r"Mean ROC (AUC = %0.2f $\pm$ %0.2f)" % (mean_auc, std_auc),
        lw=2.5,
        alpha=1)
    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    ax.fill_between(
        mean_fpr,
        tprs_lower,
        tprs_upper,
        color="grey",
        alpha=0.2)
    ax.set(
        xlim=[-0.05, 1.05],
        ylim=[-0.05, 1.05],
        ylabel="True positive rate",
        xlabel="False positive rate")
    #handles, lables = ax.get_legend_handles_labels()
    #line = Line2D([0], [0], label='manual line', color='k')
    #handles.extend([line])

    ax.legend(fontsize=15)
    #ax.get_legend().remove()
    return cm_nested