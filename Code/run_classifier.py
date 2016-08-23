# coding: utf-8

import importlib
import pandas as pd
import numpy as np
import sys

sys.path.insert(0, 'classifiers')
sys.path.insert(0, 'helpers')

from metrics import computeError, computeMetrics, computeRates, computeAUC
from add_bias import addBias
from performance import get_FPR, get_FNR, get_TPR

def run_classifier(features, labels, X_test, Y_test,
                   ## params
                   classifier,
                   ):
    try:
        classifier = importlib.import_module(classifier)
        print 'IMPORTED', classifier
    except ImportError as error:
        print error
        print "Failed to import classifier module in run_classifier.py"
        print "Available modules: 1) 'logisticReg' 2) 'adaline'"
        sys.exit(0)

    X, Y = features, labels
    
    w = classifier.fit(X, Y)
    pred = classifier.predict(X_test, w)
    
    error = computeError(Y_test, pred)
    TP, FP, FN, TN = computeMetrics(Y_test, pred)
    print 'TP:', TP
    print 'FP:', FP
    TPR, FPR, FNR, TNR = computeRates(Y_test, pred, 0, 1)

    
    AUC = computeAUC(Y_test, pred)
    return (error, TPR, FPR, FNR, TNR, AUC) 
        
# Main function to run algorithm on various fractions of attacker knowledge and control.
def main():
    df_x = pd.read_csv('../Datasets/Ham2AttackData/enron/30_perc_poison/X_train_0.csv', header = None)
    x = np.array(df_x)
    
    df_y = pd.read_csv('../Datasets/Ham2AttackData/enron/30_perc_poison/y_train_0.csv', header = None)
    y = np.array(df_y)
   
    df_x_test = pd.read_csv('../Datasets/TestData/enron/X_test_0.csv', header = None)
    x_test = np.array(df_x_test)
    
    df_y_test = pd.read_csv('../Datasets/TestData/enron/y_test_0.csv', header = None)
    y_test = np.array(df_y_test)

    classifier = 'logistic_regression'
    
    if classifier is not 'naivebayes':
        x = addBias(x)
        x_test = addBias(x_test)
    
    error, TPR, FPR, FNR, TNR, AUC = run_classifier(x, y, x_test, y_test, classifier)
    
    print 'ERROR:', error
    print 'True Positive Rate:', TPR
    print 'False Positive Rate:', FPR
    print 'False Negative Rate:', FNR
    print 'True Negative Rate:', TNR
    print 'Area Under the Curve:', AUC
    
# This is the standard boilerplate that calls the main() function.
if __name__ == '__main__':
    main()