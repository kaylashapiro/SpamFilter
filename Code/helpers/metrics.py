# coding: utf-8

import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score

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
    
    error = np.mean(np.ravel(y) != np.ravel(predictions))
    
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
    #N = y.shape[0]
    cm  = confusion_matrix(np.ravel(y), np.ravel(predictions))

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
    - FPR: false positive rate
    - TPR: true positive rate
    - thresholds: 
    '''   
    [FPR, TPR, thresholds] = roc_curve(np.ravel(y), np.ravel(predictions), drop_intermediate=False)
    
    return (FPR, TPR, thresholds)
    
def computeAUC(y, predictions):
    '''
    Returns the false positive rate and true positive rate for 
    a given training set and its predicted classes.
    
    Input:
    - y: N * 1 Numpy vector of binary feature values (0 and 1);
         class labels
    - predictions: N * 1 Numpy vector of binary values (0 and 1);
                   predicted classes
    '''
    AUC = roc_auc_score(np.ravel(y), np.ravel(predictions))
    
    return AUC
    
def computeRates(TP, FP, FN, TN):
    if ((TP + FN) == 0):
        TPR = float('nan')
        FNR = float('nan')
    else:
        TPR = float(TP)/(TP + FN)
        FNR = 1 - TPR
    if ((TN + FP) == 0):
        TNR = float('nan')
        FPR = float('nan')
    else:
        TNR = float(TN)/(TN + FP)
        FPR = 1 - TNR
    
    return (TPR, FPR, FNR, TNR)
    
def main():  
    y_true = np.array([0, 0, 1, 1])
    y_scores = np.array([0, 0, 0, 0])
    TP, FP, FN, TN = computeMetrics(y_true, y_scores)
    print computeRates(TP,FP,FN,TN)
    
    
    
    
# This is the standard boilerplate that calls the main() function.
if __name__ == '__main__':
    main()