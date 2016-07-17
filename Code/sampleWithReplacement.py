# coding: utf-8

'''
Implementation to create "bootstrap" replicates of a given training set.

A "bootstrap" replicate is a set built by random sampling with replacement 
from the original training set. 
'''

import numpy as np 
import random
import pandas as pd
import sys
    
def generateReplicate(X, y, percent_instances=1):
    '''
    Returns a single bootstrap replicate set and corresponding labels.
    
    Inputs:
    - X: no_instances * no_features Numpy matrix of binary feature values (0 and 1)
         with no_instances: the number of training examples
         and  no_features: the number of features for each example
    - y: 1 * no_instances Numpy vector of binary values (0 and 1)
    - percent_instances: float between 0 and 1; percentage of the number of instances 
                         in the training set to include in the generated bootstrap
                         replicate set
    
    Outputs:
    - X_train: new_instances * no_features Numpy matrix; bootstrap replicate
    - y_train: 1 * new_instances Numpy vector of corresponding labels
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
    - y: 1 * no_instances Numpy vector of binary values (0 and 1)
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
        
        
# Main function to create bootstrap replicate sets.
# argv[1] := number of replicates to generate.
def main():
    if len(sys.argv) >= 2:
        n_replicates = int(sys.argv[1])
    else:
        n_replicates = 25
  
    new_instances = 5
  
    # Let's get a dataset up in this bizniz
    df_X = pd.read_csv('test.csv', header = None)
    X = np.array(df_X)
    print X
    
    df_y = pd.read_csv('test_y.csv', header = None)
    y = np.array(df_y)
    print y
  
    [X, y] = generateReplicate(X, y)
    print X
    print y

# This is the standard boilerplate that calls the main() function.
if __name__ == '__main__':
    main()