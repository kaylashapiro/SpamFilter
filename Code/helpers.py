# coding: utf-8

import numpy as np

def addBias(X): 
    '''
    Adds bias term to a dataset.
    
    Input:
    
    - X: N * D Numpy matrix of binary feature values (0 and 1)
         with N: the number of training examples
         and  D: the number of features for each example
    
    Output:
    - X_bias: N * (D + 1) Numpy matrix of binary feature values
              consisting of a column of ones + X
    '''
    
    X_bias = np.insert(X, 0, 1, axis=1)
    
    return X_bias