# coding: utf-8

'''
Implementation to bag logistic regression classifiers and return their error for
a given test set and bagged predictors.

Bagging uses bootstrap replicate sets to generate different classifiers from the
same training set. These classifiers each get a vote as to which class to assign
to a new test instance.
'''

import numpy as np 
import random
import pandas as pd
import matplotlib.pyplot as plt
import logisticReg as lr
import sampleWithReplacement as swr
from sklearn.metrics import roc_curve, auc
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression

def bagFeatureSubsampling(X_train, y_train, X_test, y_test, no_predictors, percent_features=1, percent_instances=1):
    '''
    Returns array of error rates for each time a predictor is added to the
    bagged classifier. The last error rate in the bag represents the error
    rate resulting from the fully bagged classifier.
    
    Inputs:
    - X_train: N * D Numpy matrix of binary feature values (0 and 1)
               with N: the number of training examples
               and  D: the number of features for each example
    - y_train: 1 * N Numpy vector of binary values (0 and 1); training set
    - X_test: M * D Numpy matrix of binary feature values (0 and 1)
              with M: the number of test examples
    - y_test: 1 * M Numpy vector of binary values (0 and 1); test set
    - no_predictors: Number of predictors to bag
    - percent_features: Float between 0 and 1 representing the percentage of features to use in
                        bagging implemented with feature subsampling (sampling features without
                        replacement)
    
    Output:
    - errors: Array listing the error at each bagging iteration (i.e. after
              each predictor is added to the bag)
    '''
    
    no_features = X_train.shape[1]
    no_subsamples = int(round(percent_features * no_features))
    
    select_features = np.sort(np.random.choice(no_features, no_subsamples, replace=False))
    
    X_train = X_train[:,select_features]
    X_test = X_test[:,select_features]
    
    errors = bagPredictors(X_train, y_train, X_test, y_test, no_predictors)
    
    return errors
    

def bagPredictors(X_train, y_train, X_test, y_test, no_predictors, percent_instances=1):
    '''
    Returns array of error rates for each time a predictor is added to the
    bagged classifier. The last error rate in the bag represents the error
    rate resulting from the fully bagged classifier.
    
    Inputs:
    - X_train: N * D Numpy matrix of binary feature values (0 and 1); training set
               with N: the number of training examples
               and  D: the number of features for each example
    - y_train: 1 * N Numpy vector of binary values (0 and 1); training set
    - X_test: M * D Numpy matrix of binary feature values (0 and 1); test set
              with M: the number of test examples
    - y_test: 1 * M Numpy vector of binary values (0 and 1); test set
    - no_predictors: Number of predictors to bag
         
    Output:
    - errors: Array listing the error at each bagging iteration (i.e. after
              each predictor is added to the bag)
    '''
        
    errors = []

    replicate, labels = swr.generateReplicate(X_train, y_train, percent_instances)
        
    classifier_weights = lr.fit(replicate, labels)     
         
    predictions = lr.predict(X_test, classifier_weights)
   
    votes = predictions
    
    errors.append(lr.computeError(predictions, y_test))
    
    for ith_predictor in range(1, no_predictors):
        replicate, labels = swr.generateReplicate(X_train, y_train)
        
        classifier_weights = lr.fit(replicate, labels)

        votes += lr.predict(X_test, classifier_weights)
        
        predictions = computeClass(votes, ith_predictor + 1)
        
        errors.append(lr.computeError(predictions, y_test))
               
    return errors

    
def computeClass(votes, ith_predictor):
    '''
    Returns the predicted classes for a given test set and bagged classifier.
    
    Inputs:
    - votes: 1 * N array of cumulative classifications for a given test example
             with N: the number of test examples
    - ith_predictor: Number of predictors in the current bagged predictor
    
    Output:
    - votes: 1 * N array of the predicted class for a given test set  
    '''
    
    votes = votes/float(ith_predictor)
    
    votes[votes<=.5] = 0
    votes[votes>.5] = 1
    
    return votes
    

# Main function to run algorithm on various fractions of attacker knowledge and control.
def main():
    df_X = pd.read_csv('test.csv', header = None)
    X = np.array(df_X)
    print X
    
    df_y = pd.read_csv('test_y.csv', header = None)
    y = np.array(df_y)
    print y
    
    no_predictors = 3
    
    print bagFeatureSubsampling(X, y, X, y, no_predictors, percent_features=1)
    
    
# This is the standard boilerplate that calls the main() function.
if __name__ == '__main__':
    main()