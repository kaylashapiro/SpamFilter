# coding: utf-8

'''
Implementation to bag logistic regression classifiers and return their error for
a given test set and bagged predictors.

Bagging uses bootstrap replicate sets to generate different classifiers from the
same training set. These classifiers each get a vote as to which class to assign
to a new test instance.
'''

import sys
import importlib
import numpy as np 
import random
import pandas as pd
import matplotlib.pyplot as plt
import metrics as met
import sampleWithReplacement as swr
import featureSubsampling as fs
import labelSwitching as ls
from sklearn.cross_validation import train_test_split

def bagPredictors(X_train, y_train, X_test, y_test, no_predictors, 
                  perc_instances=1, perc_feature_subsampling=1, perc_label_switching=0,
                  classifier='logisticReg'):
    '''
    Returns array of error rates for each time a predictor is added to the
    bagged classifier. The last error rate in the bag represents the error
    rate resulting from the fully bagged classifier.
    
    Inputs:
    - X_train: N * D Numpy matrix of binary feature values (0 and 1); training set
               with N: the number of training examples
               and  D: the number of features for each example
    - y_train: N * 1 Numpy vector of binary values (0 and 1); training set
    - X_test: M * D Numpy matrix of binary feature values (0 and 1); test set
              with M: the number of test examples
    - y_test: M * 1 Numpy vector of binary values (0 and 1); test set
    - no_predictors: Number of predictors to bag
    - perc_instances: Float between 0 and 1 representing the percentage of instances to include
                      in each bootstrap replicate set based on the number of instances in the 
                      training set
    - perc_feature_subsampling: Float between 0 and 1 representing the percentage of features to
                                use in bagging implemented with feature subsampling (sampling 
                                features without replacement)
    - perc_label_switching: Float between 0 and 1 representing the percentage of labels to switch
                            (flip) in the training set
    - classifier: default string classifier name; Options: 1) 'logisticReg' 2) 'adaline'
         
    Output:
    - errors: 1 * no_predictors array listing the error at each bagging iteration 
              (i.e. after each predictor is added to the bag)
    '''
    try:
        classifier = importlib.import_module(classifier)
    except ImportError as error:
        print error
        print "Failed to import classifier module in bagging.py"
        print "Available modules: 1) 'logisticReg' 2) 'adaline'"
        sys.exit(0)
    
    errors = []
    TPRs = []
    FPRs = []
    FNRs = []
    TNRs = []
    AUCs = []
    if (perc_feature_subsampling != 1):
        X_train, X_test = fs.featureSubsampling(X_train, X_test, perc_feature_subsampling)
    if (perc_label_switching != 0):
        y_train = ls.labelSwitching(y_train, perc_label_switching)
    replicate, labels = swr.generateReplicate(X_train, y_train, perc_instances)
    
    classifier_weights = classifier.fit(replicate, labels)     
         
    votes = classifier.predict(X_test, classifier_weights)
   
    predictions = votes
    
    errors.append(met.computeError(y_test, predictions))
    
    AUCs.append(met.computeAUC(y_test, predictions))
    
    [TP, FP, FN, TN] = met.computeMetrics(y_test, predictions)
    
    [TPR, FPR, FNR, TNR] = met.computeRates(TP, FP, FN, TN)
    
    TPRs.append(TPR)
    FPRs.append(FPR)
    FNRs.append(FNR)
    TNRs.append(TNR)
    
    for ith_predictor in range(1, no_predictors):
        if (ith_predictor%10 == 0):
            print 'Predictor:', ith_predictor
            
        replicate, labels = swr.generateReplicate(X_train, y_train)
        
        classifier_weights = classifier.fit(replicate, labels)

        votes += classifier.predict(X_test, classifier_weights)
        
        predictions = computeClass(votes, ith_predictor + 1)
        
        errors.append(met.computeError(y_test, predictions))
        
        AUCs.append(met.computeAUC(y_test, predictions))
        
        [TP, FP, FN, TN] = met.computeMetrics(y_test, predictions)
    
        [TPR, FPR, FNR, TNR] = met.computeRates(TP, FP, FN, TN)
    
        TPRs.append(TPR)
        FPRs.append(FPR)
        FNRs.append(FNR)
        TNRs.append(TNR)
               
    return (errors, TPRs, FPRs, FNRs, TNRs, AUCs)

    
def computeClass(votes, ith_predictor):
    '''
    Returns the predicted classes for a given test set and bagged classifier.
    
    Inputs:
    - votes: N * 1 array of cumulative classifications for a given test example
             with N: the number of test examples
    - ith_predictor: Number of predictors in the current bagged predictor
    
    Output:
    - votes: N * 1 array of the predicted class for a given test set  
    '''
    
    votes = votes/float(ith_predictor)
    
    votes[votes<=.5] = 0
    votes[votes>.5] = 1
    
    return votes
    

# Main function to run algorithm on various fractions of attacker knowledge and control.
def main():
    df_X = pd.read_csv('../Datasets/EmailDataProcessed/Features.csv', header = None)
    X = np.array(df_X)
    print X
    
    df_y = pd.read_csv('../Datasets/EmailDataProcessed/Labels.csv', header = None)
    y = np.array(df_y)
    print y
    
    no_predictors = 3
    
    print bagPredictors(X, y, X, y, no_predictors)
    
    
# This is the standard boilerplate that calls the main() function.
if __name__ == '__main__':
    main()