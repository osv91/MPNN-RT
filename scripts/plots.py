import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np


#plotting functions will be applied only for train-test splits
#not for CV-mode


def plot_learning_curve(history, figures_path, dataset_name):
        plt.figure(figsize=(10, 6))
        plt.plot(history.history["MAE"], label="train AUC")
        plt.plot(history.history["val_MAE"], label="valid AUC")
        plt.xlabel("Epochs", fontsize=16)
        plt.ylabel("AUC", fontsize=16)
        plt.legend(fontsize=16)
        plt.savefig(figures_path+ '/'+ 'CV_learning_curve_'+dataset_name+".svg")
        
def plot_pred_exp(Y_true, y_pred, figures_path, dataset_name):
    
    plt.figure(figsize=(10, 6))
    plt.scatter(Y_true, y_pred, c=np.random.uniform(0, 0.25, Y_true.shape[0]), cmap='viridis')
    plt.xlabel("Experimental RT, s", fontsize=16)
    plt.ylabel("Predicted RT, s", fontsize=16)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.savefig(figures_path+ '/'+ 'Exp-Pred_'+dataset_name+".svg")
    
    
def plot_err_dist(Y_true, y_pred, figures_path, dataset_name):
    ae = y_pred-Y_true
    plt.figure(figsize=(10, 6))
    sns.displot(ae, bins=40)
    plt.xlabel("Absolute error, s", fontsize=16)
    plt.ylabel("", fontsize=16)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.savefig(figures_path+ '/'+ 'AE_dist_'+dataset_name+".svg")
    
def MRE_boxplot(Y_true, y_pred, figures_path, dataset_name):
    ae = 100*(y_pred-Y_true)/Y_true
    plt.figure(figsize=(10, 6))
    bplot = plt.boxplot(ae, patch_artist=True)
    for patch in bplot['boxes']:
        patch.set_facecolor('lightgreen')
        
    plt.xlabel("", fontsize=16)
    plt.ylabel("Mean relative error, %", fontsize=16)
    #plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.savefig(figures_path+ '/'+ 'Boxplot_MRE_'+dataset_name+".svg")