# coding: utf-8

'''
Implementation of a simple focused attack. 

Injects copies of same malicious data point.

Assumes no bias has been added yet.
'''

import numpy as np
import random
import pandas as pd

def selectHamIndex(y):
    '''
    Returns the index of a random ham email in the training set.
    
    Inputs:
    - y: 1 * no_instances Numpy vector of binary values (0 and 1) 
         with no_instances: the number of training examples
    
    Output: 
    - ham_email: index of random ham email in the training set
    '''
    
    ham_index = [i for i in range(len(y)) if y[i] == 0]
    
    ham_email = np.random.choice(ham_index, 1)
    
    return ham_email

    
def generateAttackData(ham_email, features_present, no_mal_instances, no_features, no_mal_features):   
    '''
    Returns crafted attack instances.
    
    Inputs:
    - ham_email: 1 * no_features Numpy vector with features corresponding to one random
                 ham email from the training set.
    - features_present: list of indices where feature value in ham_email is set to 1
    - no_mal_instances: number of poison data examples
    - no_features: the number of features for each example
    - no_mal_features: the number of features the attacker knows about
    
    Output: 
    - mal_data: no_mal_instances * no_features Numpy matrix of poison examples
    
    EXTENSION: Implement more sophisticated attacker knowledge rather than random
    EXTENSION: Implement attack with different (varied) data points
    '''
    
    rand_features = np.array([0] * (no_features - no_mal_features) + [1] * no_mal_features)
    np.random.shuffle(rand_features)
    
    mal_ham = np.array(ham_email)
    
    mal_ham[features_present] = rand_features
  
    mal_data = np.array([mal_ham,] * no_mal_instances)
      
    return mal_data
    
    
def poisonData(X, y, frac_knowl, frac_mal_instances):
    '''
    Returns the input data with *added* data that is crafted specifically to cause
    a poisoning focused attack, where all features (within attacker knowledge)
    corresponding to a specific ham email are set to 1 and marked as spam. 
    
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
    
    ham_email = np.array(X[selectHamIndex(y)][0])
    
    features_present = [i for i in range(len(ham_email)) if ham_email[i] == 1]
    
    no_features = len(features_present)

    no_instances = X.shape[0]
    
    no_mal_features = int(round(frac_knowl * no_features))
    no_mal_instances = int(round(frac_mal_instances * no_instances))
        
    mal_data = generateAttackData(ham_email, features_present, no_mal_instances, no_features, no_mal_features)
    mal_y = np.ones(no_mal_instances, dtype=np.int) # Contamination assumption
    
    indices = np.random.choice(X.shape[0], no_mal_instances, replace=False)
    
    X[indices] = mal_data
    y[indices] = 1
    
    return (X, y)
    

# Main function to run the dictionary attack
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