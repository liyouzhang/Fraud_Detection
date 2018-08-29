import numpy as np
from sklearn.metrics import roc_curve, auc, confusion_matrix
import matplotlib.pyplot as plt

def standard_confusion_matrix(y_true, y_pred):
    [[tn, fp], [fn, tp]] = confusion_matrix(y_true, y_pred)
    return np.array([[tp, fp], [fn, tn]])

def profit_curve(cost_benefit, predicted_probs, labels):
    '''

    Arguments:
        cost_benefit {[numpy matrix]} -- a matrix to assign dollar values to TP,FP, FN and TN
        predicted_probs {[numpy array]} -- predicted probability
        labels {[numpy array]} -- y_true

    Returns:
        profits[list] -- a list of list
    '''

    n_obs = float(len(labels))
    thresholds = np.arange(0,1,0.01)
    profits = []
    for threshold in thresholds:
        y_predict = predicted_probs >= threshold
        confusion_matrix = standard_confusion_matrix(labels, y_predict)
        threshold_profit = np.sum(confusion_matrix * cost_benefit) / n_obs
        profits.append([threshold_profit,threshold])
    return profits

def plot_model_profits(profits, save_path=None):
    '''plot the profit curve
    
    Arguments:
        profits {[list]} -- the output of profit_curve function
    
    Keyword Arguments:
        save_path {[str]} -- if None, then don't save; otherwise it's the pathway to save (default: {None})
    '''

    threshold = []
    profit = []
    for p in profits:
        threshold.append(p[1])
        profit.append(p[0])
    plt.figure(figsize=(4,3))
    plt.plot(threshold, profit)
    plt.ylim(0,8.5)
    plt.title("Profit Curve")
    plt.xlabel("TPR-FPR Threshold")
    plt.ylabel("Profit ($/user)")
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()