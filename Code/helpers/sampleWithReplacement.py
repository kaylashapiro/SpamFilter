# coding: utf-8

'''
Implementation to create "bootstrap" replicates of a given training set.

A "bootstrap" replicate is a set built by random sampling with replacement 
from the original training set. 
'''

import numpy as np 
import random
import sys
    
def generateReplicate(X, y, percent_instances=1):
    '''
    Returns a single bootstrap replicate set and corresponding labels.
    
    Inputs:
    - X: no_instances * no_features Numpy matrix of binary feature values (0 and 1)
         with no_instances: the number of training examples
         and  no_features: the number of features for each example
    - y: no_instances * 1 Numpy vector of binary values (0 and 1)
    - percent_instances: float between 0 and 1; percentage of the number of instances 
                         in the training set to include in the generated bootstrap
                         replicate set
    
    Outputs:
    - X_train: new_instances * no_features Numpy matrix; bootstrap replicate
    - y_train: new_instances * 1 Numpy vector of corresponding labels
    '''
    
    new_instances = int(round(percent_instances*X.shape[0]))
    
    indices = np.random.choice(X.shape[0], new_instances, replace=True)
    
    X_train = np.array(X[indices])
    y_train = y[indices]
     
    return (X_train, y_train)
    

def generateBootstraps(X, y, n_replicates, percent_instances=1):
    '''
    Generates a certain number of replicate sets and saves them to .csv file
    for a specified path.
    
    Inputs:
    - X: no_instances * no_features Numpy matrix of binary feature values (0 and 1)
         with no_instances: the number of training examples
         and  no_features: the number of features for each example
    - y: no_instances * 1 Numpy vector of binary values (0 and 1)
    - n_replicates: number of replicate sets to generate
    - percent_instances: float between 0 and 1; percentage of the number of instances 
                         in the training set to include in the generated bootstrap
                         replicate set; default set to 1
    
    Output:
    NONE
    '''
    folder = './Bootstraps/'

    for i in range(0, n_replicates):
        replicate, labels = generateReplicate(X, y, percent_instances)
        
        filename = 'replicate' + str(i+1) + '.csv'
        out_name = folder + filename
        
        with open(out_name, 'wb') as ofile:
            np.savetxt(ofile, replicate, fmt='%u', delimiter=',')
        