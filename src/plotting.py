
from sklearn.metrics import RocCurveDisplay, ConfusionMatrixDisplay
from sklearn.metrics import auc

import numpy as np
import seaborn as sn



def nested_ROC_plot(y_test_nested, y_pred_nested, ax=None):
    """
    Plot ROC curve of each outer layer from nested crossvalidation
    with the standard deviation.

    Parameters
    ----------
    y_test_nested: Estimated targets as returned by a classifier.
         Ground truth (correct) target values.
         y_test splits used in nested cv.

    y_pred_nested: Estimated targets as returned by a classifier.
        Desicion function of the classifier.

    ax: matplotlib axes, default=None
        Axes object to plot on. If None, a new figure and axes is created.
    
    Return
    ------
     viz: RocCurveDisplay
        Object that stores computed values
    """

    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)
    sn.set_style("ticks")
    sn.set_context("talk", font_scale=1.5)
    
    # create axis
    ax=ax

    for y_test, y_pred in zip(y_test_nested, y_pred_nested):

    # ROC curve from predictions
        viz = RocCurveDisplay.from_predictions(
                y_test,
                y_pred,
                alpha=0.4,
                lw=1,
                #label="_",
                ax=ax)


        interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
        aucs.append(viz.roc_auc)

    # Adding dumb classifier and standard deviation 
    
    ax.plot([0, 1], [0, 1], linestyle="--", lw=2, color="r", alpha=0.8)
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)


    ax.plot(
        mean_fpr,
        mean_tpr,
        color="darkblue",
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


    #ax.legend(fontsize=15)
    ax.get_legend().remove()
    print(f"Mean AUC = {mean_auc:.3f} ({std_auc:.3f})")
    return viz

def feature_importance_plot(features_importance, ylabels=None, ax=None):
    """
    Plot variable importance/coefficients vs wavenumbers
    
    Parameters
    ----------
    features_importance: dataframe
    ylabels: string

    Return
    ------ 
    """

    sn.set_theme(style="ticks")
    sn.color_palette("tab10")
    sn.set_context("notebook", font_scale=1.2)

    ax = ax
    ax.plot(features_importance.iloc[:,0], features_importance.iloc[:,1], color="darkgreen")
    ax.set_xlim(1800,600)
    ax.set_xlabel('Wavelenght (cm$^{-1}$)', weight='bold')
    ax.set_ylabel(ylabels, weight='bold')
    ax.axhline(0, ls='--',color='k')


def confusion_matrix_plotting(y_true, y_pred, ax=None):
    """
    Plot a confusion matrix using RocCurveDisplay object.

    Parameters
    ----------
    y_true: list 
         Ground truth (correct) target values.

    y_pred: list
         Estimated targets as returned by a classifier.

    ax: matplotlib axes, default=None
        Axes object to plot on. If None, a new figure and axes is created.
    
    Return
    ------
     disp: RocCurveDisplay
        Object that stores computed values
    """

    sn.set_context("talk", font_scale=1.2)
    ax = ax
    disp = ConfusionMatrixDisplay.from_predictions(y_true,
                                                   y_pred,
                                                   normalize='true',
                                                   cmap='PuBu',
                                                   im_kw={'vmin':0, 'vmax':1},
                                                   ax=ax)
    return disp


def confusion_matrix_nested(cm_nested, labels, ax=None):

    """ Plot a confusion matrix using RocCurveDisplay object.

    Parameters
    ----------
    cm_nested : ndarray of shape (n_classes, n_classes)
        Confusion matrix whose i-th row and j-th
        column entry indicates the number of
        samples with true label being i-th class
        and predicted label being j-th class.

    ax: matplotlib axes, default=None
        Axes object to plot on. If None, a new figure and axes is created.
    
    Return
    ------
     disp: RocCurveDisplay
        Object that stores computed values
    """

    # normalization
    cm_nested_normalized = cm_nested.astype('float')/cm_nested.sum(axis=1)[:, np.newaxis]

    # figure
    sn.set_context("talk", font_scale=1.1)
    ax = ax
    disp = ConfusionMatrixDisplay(cm_nested_normalized, display_labels=labels)
    disp.plot(cmap = "PuBu",
              im_kw={'vmin':0, 'vmax':1},ax=ax, colorbar=False)
    #plt.xticks(rotation=45)
    #plt.xticks([0.5,1.5],label_names, ha = 'center')
    #plt.yticks([0.5,1.5],label_names)
    return disp