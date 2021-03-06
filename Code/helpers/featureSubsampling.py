# coding: utf-8

import numpy as np 

def featureSubsampling(X_train, X_test, perc_feature_subsampling):
    '''
    Returns training set and test set selecting on random features of a certain
    percentage based on the original feature set size.
    
    Inputs:
    - X_train: N * D Numpy matrix of binary feature values (0 and 1); training set
               with N: the number of training examples
               and  D: the number of features for each example
    - X_test: M * D Numpy matrix of binary values (0 and 1); test set 
              with M: the number of test examples
    - perc_feature_subsampling: Real number between 0 and 1 representing the percentage
                                of features to use in a given feature set (sampling 
                                features without replacement)
    
    Outputs:
    - X_train: feature training set with selected features
    - X_test: feature test set with selected features
    '''
    no_features = X_train.shape[1]
    no_subsamples = int(round(perc_feature_subsampling * no_features))
    
    select_features = np.sort(np.random.choice(no_features, no_subsamples, replace=False))
    
    X_train = X_train[:,select_features]
    X_test = X_test[:,select_features]
    
    return (X_train, X_test)
