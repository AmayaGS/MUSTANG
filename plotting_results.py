# -*- coding: utf-8 -*-
"""
Created on Thu Feb  2 18:50:47 2023

@author: AmayaGS
"""

import os, os.path
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
import itertools

from PIL import Image
from PIL import ImageFile

import numpy as np

from matplotlib import pyplot as plt

from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc
from sklearn.preprocessing import label_binarize
from sklearn.metrics import auc as calc_auc

Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True

plt.ion()


def auc_plot(labels, prob):

    # AUC

    fpr, tpr, _ = roc_curve(labels, prob)
    plt.figure(figsize=(10,10))
    plt.plot(fpr, tpr, color='r')
    plt.plot([0, 1], [0, 1], color = 'black', linestyle = '--')
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.xlabel('False Positive Rate', size=18)
    plt.ylabel('True Positive Rate', size=18)
    #plt.title('AUC = %0.2f' % test_auc, size=20)
    #plt.text(0.3, 0.0, 'Sensitivity = 0.90\nSpecificity = 0.86', fontsize = 20)
    plt.savefig('auroc.png')

def pr_plot(labels, prob):

    # PR

    precision, recall, thresholds = precision_recall_curve(labels, prob)
    auc_precision_recall = auc(recall, precision)
    plt.figure(figsize=(10,10))
    plt.plot(recall, precision, color='darkblue')
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    #plt.title('Average Precision = %0.2f' % auc_precision_recall, size=20)
    plt.ylabel('Precision', size=18)
    plt.xlabel('Recall', size=18)
    plt.legend(loc='lower left', prop={'size': 19})
    plt.savefig('prc.png')

def confusion_matrix_plot(cm,
                          target_names,
                          title='Confusion matrix',
                          cmap=None,
                          normalize=True):
    """
    given a sklearn confusion matrix (cm), make a nice plot

    Arguments
    ---------
    cm:           confusion matrix from sklearn.metrics.confusion_matrix

    target_names: given classification classes such as [0, 1, 2]
                  the class names, for example: ['high', 'medium', 'low']

    title:        the text to display at the top of the matrix

    cmap:         the gradient of the values displayed from matplotlib.pyplot.cm
                  see http://matplotlib.org/examples/color/colormaps_reference.html
                  plt.get_cmap('jet') or plt.cm.Blues

    normalize:    If False, plot the raw numbers
                  If True, plot the proportions

    Usage
    -----
    plot_confusion_matrix(cm           = cm,                  # confusion matrix created by
                                                              # sklearn.metrics.confusion_matrix
                          normalize    = True,                # show proportions
                          target_names = y_labels_vals,       # list of names of the classes
                          title        = best_estimator_name) # title of graph

    Citiation
    ---------
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

    """

    accuracy = np.trace(cm) / np.sum(cm).astype('float')
    #misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title('Accuracy={:0.2f}'.format(accuracy), size=25)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=0, size=20)
        plt.yticks(tick_marks, target_names, rotation=0, size=20)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


    thresh = cm.max() / 1.4 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            # plt.text(j, i, "{:0.2f}".format(cm[i, j]),
            #          horizontalalignment="center",
            #          color="white" if cm[i, j] > thresh else "black")
            plt.text(j, i, "{:0.2f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="black", size=25)
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
    plt.text(j, i, "{:0.2f}".format(cm[1, 1]),
             horizontalalignment="center",
             color="white", size=25)


    plt.tight_layout()
    plt.ylabel('True label', size=20)
    plt.xlabel('Predicted label', size=20)
    plt.show()


def learning_curve_plot(df_results):

    plt.rcParams.update({'font.size': 20})
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(25, 10))
    df_results["train_accuracy"].plot(ax=axes[0], color='r')
    df_results["val_accuracy"].plot(ax=axes[0], color='b')
    axes[0].set_xticks(range(0,len(df_results["train_accuracy"])))
    axes[0].set_xlabel("Epochs")
    axes[0].set_ylabel("Accuracy")
    df_results["train_loss"].plot(ax=axes[1], color='orange')
    df_results["val_loss"].plot(ax=axes[1], color='green')
    axes[1].set_xticks(range(0,len(df_results["train_loss"])))
    axes[1].set_xlabel("Epochs")
    axes[1].set_ylabel("Loss")
    axes[0].legend(loc='best')
    axes[1].legend(loc='best')
    plt.show()
    fig.savefig("learning_curve.png", bbox_inches='tight')