# coding: utf-8

import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve, auc

def computeError(y, predictions):
    '''
    Returns the average error for a given training set and its
    predicted classes.
    
    Input:
    - y: N * 1 Numpy vector of binary feature values (0 and 1);
         class labels
    - predictions: N * 1 Numpy vector of binary values (0 and 1);
                   predicted classes
    
    Output:
    - error: float average error
    '''
    
    error = np.mean(y != predictions)
    
    return error
    
def computeMetrics(y, predictions):
    '''
    Returns the number of true positives, false positives, false
    negatives, and true negatives for a given training set and its
    predicted classes.
    
    Input:
    - y: N * 1 Numpy vector of binary feature values (0 and 1);
         class labels
    - predictions: N * 1 Numpy vector of binary values (0 and 1);
                   predicted classes
    
    Output:
    - TP: number of true positives
    - FP: number of false positives
    - FN: number of false negatives
    - TN: number of true negatives
    '''
    cm  = confusion_matrix(y, predictions)

    TP = cm[0][0]
    FP = cm[0][1]
    FN = cm[1][0]
    TN = cm[1][1]
    
    return (TP, FP, FN, TN)
    
def computeROC(y, predictions, pos_label=1):
    '''
    Returns the false positive rate and true positive rate for 
    a given training set and its predicted classes.
    
    Input:
    - y: N * 1 Numpy vector of binary feature values (0 and 1);
         class labels
    - predictions: N * 1 Numpy vector of binary values (0 and 1);
                   predicted classes
    
    Output:
    - FPR: 
    - TPR: 
    - thresholds
    '''
    
    [FPR, TPR, thresholds] = roc_curve(y, predictions, drop_intermediate=False)
    
    return (FPR, TPR, thresholds)
    
def computeRates(TP, FP, FN, TN):
    TPR = float(TP)/(TP + FN)
    FNR = 1 - TPR
    TNR = float(TN)/(TN + FP)
    FPR = 1 - TNR
    
    return (TPR, FPR, FNR, TNR)