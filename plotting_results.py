# -*- coding: utf-8 -*-
"""
Created on Thu Feb  2 18:50:47 2023

@author: AmayaGS
"""

import os, os.path
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
import itertools
import numpy as np

from PIL import Image
from PIL import ImageFile

from matplotlib import pyplot as plt

from sklearn.metrics import roc_curve
from sklearn.metrics import precision_recall_curve

Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True

plt.ion()

class plot_train_results():

    def __init__(self, df_results, save="results/"):

        self.df_results = df_results
        self.save = save

    def learning_curve_plot(self):

        plt.rcParams.update({'font.size': 20})
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(25, 10))
        self.df_results["train_accuracy"].plot(ax=axes[0], color='r')
        self.df_results["val_accuracy"].plot(ax=axes[0], color='b')
        axes[0].set_xticks(range(0,len(self.df_results["train_accuracy"])))
        axes[0].set_xlabel("Epochs")
        axes[0].set_ylabel("Accuracy")
        self.df_results["train_loss"].plot(ax=axes[1], color='orange')
        self.df_results["val_loss"].plot(ax=axes[1], color='green')
        axes[1].set_xticks(range(0,len(self.df_results["train_loss"])))
        axes[1].set_xlabel("Epochs")
        axes[1].set_ylabel("Loss")
        axes[0].legend(loc='best')
        axes[1].legend(loc='best')
        plt.show()
        fig.savefig(self.save + "learning_curve.png", bbox_inches='tight')

    def plot(self):

        plot_train_results.learning_curve_plot(self)


class plot_test_results():

    def __init__(self, labels, prob, conf_matrix, target_names=["Class 1", "Class 2"], save="results/"):

        self.labels = labels
        self.prob = prob
        self.save = save
        self.cm = conf_matrix
        self.target_names = target_names

    def auc_plot(self):
        # AUC
        fpr, tpr, _ = roc_curve(self.labels, self.prob[:, 1])
        plt.figure(figsize=(10,10))
        plt.plot(fpr, tpr, color='r')
        plt.plot([0, 1], [0, 1], color = 'black', linestyle = '--')
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.xlabel('False Positive Rate', size=18)
        plt.ylabel('True Positive Rate', size=18)
        #plt.title('AUC = %0.2f' % test_auc, size=20)
        #plt.text(0.3, 0.0, 'Sensitivity = 0.90\nSpecificity = 0.86', fontsize = 20)
        plt.savefig(self.save + 'auroc.png')

    def pr_plot(self):
        # PR
        precision, recall, thresholds = precision_recall_curve(self.labels, self.prob[:, 1])
        #auc_precision_recall = auc(recall, precision)
        plt.figure(figsize=(10,10))
        plt.plot(recall, precision, color='darkblue')
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        #plt.title('Average Precision = %0.2f' % auc_precision_recall, size=20)
        plt.ylabel('Precision', size=18)
        plt.xlabel('Recall', size=18)
        plt.legend(loc='lower left', prop={'size': 19})
        plt.savefig(self.save + 'prc.png')

    def confusion_matrix_plot(self,
                              title='Confusion matrix',
                              cmap=None,
                              normalize=True,
                              save= "results/"):

        accuracy = np.trace(self.cm) / np.sum(self.cm).astype('float')
        #misclass = 1 - accuracy

        if cmap is None:
            cmap = plt.get_cmap('Blues')

        plt.figure(figsize=(8, 6))
        plt.imshow(self.cm, interpolation='nearest', cmap=cmap)
        plt.title('Accuracy={:0.2f}'.format(accuracy), size=25)
        plt.colorbar()

        if self.target_names is not None:
            tick_marks = np.arange(len(self.target_names))
            plt.xticks(tick_marks, self.target_names, rotation=0, size=20)
            plt.yticks(tick_marks, self.target_names, rotation=0, size=20)

        if normalize:
            self.cm = self.cm.astype('float') / self.cm.sum(axis=1)[:, np.newaxis]


        thresh = self.cm.max() / 1.4 if normalize else self.cm.max() / 2
        for i, j in itertools.product(range(self.cm.shape[0]), range(self.cm.shape[1])):
            if normalize:
                # plt.text(j, i, "{:0.2f}".format(self.cm[i, j]),
                #          horizontalalignment="center",
                #          color="white" if self.cm[i, j] > thresh else "black")
                plt.text(j, i, "{:0.2f}".format(self.cm[i, j]),
                         horizontalalignment="center",
                         color="black", size=25)
            else:
                plt.text(j, i, "{:,}".format(self.cm[i, j]),
                         horizontalalignment="center",
                         color="white" if self.cm[i, j] > thresh else "black")
        plt.text(j, i, "{:0.2f}".format(self.cm[1, 1]),
                 horizontalalignment="center",
                 color="white", size=25)

        plt.tight_layout()
        plt.ylabel('True label', size=20)
        plt.xlabel('Predicted label', size=20)
        plt.savefig(self.save + "conf_matrix.png", bbox_inches='tight')

    def plot(self):

        plot_test_results.auc_plot(self)
        plot_test_results.pr_plot(self)
        plot_test_results.confusion_matrix_plot(self)