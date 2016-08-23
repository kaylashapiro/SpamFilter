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
    
def computeRates(Y, O, ham_label, spam_label):
    
    TPR = get_TPR(Y, O, ham_label, spam_label)
    FPR = get_FPR(Y, O, ham_label, spam_label)
    FNR = 1 - TPR
    TNR = 1 - FPR
    
    return (TPR, FPR, FNR, TNR)

def get_FPR(Y, O, ham_label, spam_label):
    '''
    Calculates False Positive Rate (=fall-out), also called false alarm rate
    
    Input:
    - Y: N * 1 Numpy vector of binary values (0 and 1); class labels
    - O: N * 1 Numpy vector of binary values (0 and 1); predicted class labels
    
    Output:
    - FPR: float, false positive rate
    '''
    Y, O = map(np.ravel, [Y, O]) ## make sure shape is (len,) for both
    FP = np.sum((O == spam_label) & (Y == ham_label))
    N =  np.sum(Y == ham_label) ## FP + TN
    FPR = float(FP) / N
    return FPR

def get_TPR(Y, O, ham_label, spam_label):
    '''
    Calculates True Positive Rate
    
    Input:
    - Y: N * 1 Numpy vector of binary values (0 and 1); class labels
    - O: N * 1 Numpy vector of binary values (0 and 1); predicted class labels
    
    Output:
    - TNR: float, true negative rate
    '''
    Y, O = map(np.ravel, [Y, O]) ## make sure shape is (len,) for both
    TP = np.sum((O == spam_label) & (Y == spam_label))
    P =  np.sum(Y == spam_label) ## TP + FN
    TPR = float(TP) / P
    return TPR
    
def main():  
    y_true = np.array([0, 0, 1, 1])
    y_scores = np.array([0, 0, 0, 0])
    print computeRates(y_true, y_scores, 0, 1)
    
    
# This is the standard boilerplate that calls the main() function.
if __name__ == '__main__':
    main()