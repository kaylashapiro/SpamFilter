# coding: utf-8

'''
Implementation of a simple dictionary attack. 

Injects copies of same malicious data point.

Assumes no bias has been added yet.
'''

import numpy as np
import random
import pandas as pd
import logisticRegVec as reg

def generateAttackData(no_mal_instances, no_features, no_mal_features):
    '''
    Returns crafted attack instances.
    
    Inputs:
    - no_mal_instances: number of poison data examples
    - no_features: the number of features for each example
    - no_mal_features: the number of features the attacker knows about
    
    Output: 
    - mal_data: set of poison examples
    
    TODO: Think about implementing more sophisticated attacker knowledge rather than random
    TODO: Think about implementing attack with different data points
    '''
    
    rand_features = np.array([0] * (no_features - no_mal_features) + [1] * no_mal_features)
    np.random.shuffle(rand_features)
    
    mal_data = np.array([rand_features,] * no_mal_instances)
    
    return mal_data
    

def poisonData(X, y, frac_knowl, frac_mal_instances):
    '''
    Returns the input data with *added* data that is crafted specifically to cause
    a poisoning dictionary attack, where all features (within attacker knowledge)
    are set to 1. 
    
    Inputs:
    - X: no_instances * no_features Numpy matrix of binary feature values (0 and 1)
        with no_instances: the number of training examples
        and  no_features: the number of features for each example
    - y: 1 * no_instances Numpy vector of binary values (0 and 1)
    - frac_knowl: float between 0 and 1
                  percentage of knowledge the attacker has of the feature set
    - frac_mal_instances: float between 0 and 1
                          percentage of the dataset under the attacker's control
    
    Outputs:
    - X: poisoned features
    - y: poisoned labels    
    '''
    
    no_instances, no_features = X.shape
    
    no_mal_features = int(round(frac_knowl * no_features))
    no_mal_instances = int(round(frac_mal_instances * no_instances))
        
    mal_data = generateAttackData(no_mal_instances, no_features, no_mal_features)
    mal_y = np.ones(no_mal_instances, dtype=np.int) # Contamination assumption
    
    indices = np.random.choice(X.shape[0], no_mal_instances, replace=False)
    
    X[indices] = mal_data
    y[indices] = 1
    
    return (X, y)
    


# Main function to test the dictionary attack
def main():
    df_X = pd.read_csv('test.csv', header = None)
    X = np.array(df_X)
    print X
    
    df_y = pd.read_csv('test_y.csv', header = None)
    y = np.array(df_y).T[0]
    print y
    
    frac_knowl = .5
    frac_mal_instances = 1.0/3
    
    print poisonData(X, y, frac_knowl, frac_mal_instances)
    
    

# This is the standard boilerplate that calls the main() function.
if __name__ == '__main__':
    main()