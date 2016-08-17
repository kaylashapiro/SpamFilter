# coding: utf-8

'''
Implementation of a simple focused attack. 

Injects copies of same malicious data point.

Assumes no bias has been added yet.
'''

import numpy as np
import random
import pandas as pd

def select_ham_index(labels):
    '''
    Returns the index of a random ham email in the training set.
    
    Inputs:
    - labels: 1 * no_instances Numpy vector of binary values (0 and 1) 
        with no_instances: the number of training examples
    
    Output: 
    - ham_index: index of random ham email in the training set
    '''
    Y = labels
    
    ## Find all the instances of ham
    ham_indices = [i for i in range(len(Y)) if Y[i] == 0]
    
    ## Select a random ham index
    ham_index = np.random.choice(ham_indices, 1)
    
    return ham_index

    
def generateAttackData(ham_email, features_present, no_poisoned, d, no_poisoned_features):   
    '''
    Returns crafted attack instances.
    
    Inputs:
    - ham_email: 1 * no_features Numpy vector with features corresponding to one random
                 ham email from the training set.
    - features_present: list of indices where feature value in ham_email is set to 1
    - no_poisoned: number of poison data examples
    - d: the number of features to "turn on" for each example
    - no_poisoned_features: the number of features the attacker knows about
    
    Output: 
    - attack_points: no_poisoned * no_features Numpy matrix of poison examples
    
    EXTENSION: Implement more sophisticated attacker knowledge rather than random
    EXTENSION: Implement attack with different (varied) data points
    '''
    
    rand_features = np.array([0] * (d - no_poisoned_features) + [1] * no_poisoned_features)
    np.random.shuffle(rand_features)
    
    attack_point = np.array(ham_email)
    
    attack_point[features_present] = rand_features
  
    attack_points = np.array([attack_point,] * no_poisoned)
      
    return attack_points
    
    
def poisonData(features, labels, 
               ## params
               percentage_samples_poisoned,
               percentage_features_poisoned=1.0,
               feature_selection_method=None,
               threshold=.1,
               ham_email=None,
               ham_index=None,
               ham_label=0,
               spam_label=1,
               ):
    '''
    Returns the input data with *added* data that is crafted specifically to cause
    a poisoning focused attack, where all features (within attacker knowledge)
    corresponding to a specific ham email are set to 1 and marked as spam. 
    
    Inputs:
    - features: N * D Numpy matrix of binary feature values (0 and 1)
                with N: the number of training examples
                and  D: the number of features for each example
    - labels: N * 1 Numpy vector of binary values (0 and 1)
    - percentage_samples_poisoned: float between 0 and 1
                percentage of the dataset under the attacker's control
    - percentage_features_poisoned: float between 0 and 1
                percentage of knowledge the attacker has of the feature set
    - feature_selection_method: string
    - threshold: percentile of features to keep
    
    Outputs:
    - X: poisoned features
    - Y: poisoned labels    
    '''
    ## notation
    X, Y = features, labels
    N, D = X.shape
    
    if not ham_email:      
        if not ham_index:
            ## select a random ham instance from the training set
            ham_index = select_ham_index(Y)
    
        ham_email = np.array(X[ham_index][0])
    
    ## find the indices of features that are "turned on" in the ham instance
    features_present = [i for i in range(len(ham_email)) if ham_email[i] == 1]
    
    ## number of features "turned on"
    d = len(features_present)

    ## number of features to poison
    no_poisoned_features = int(round(percentage_features_poisoned * d))
    
    ## number of attack points to inject
    no_poisoned = int(round(percentage_samples_poisoned * N))
        
    attack_points = generateAttackData(ham_email, features_present, no_poisoned, d, no_poisoned_features)
    
    poisoned_indices = np.random.choice(N, no_poisoned, replace=False)
    
    X[poisoned_indices] = attack_points
    Y[poisoned_indices] = spam_label
    
    return (X, Y)
    
# Main function to test the ham attack
def main():
    x = np.array([[1, 0, 1],		
        [0, 0, 0],		
        [1, 0, 1],		
        [1, 1, 1],		
        [1, 1, 0],		
        [1, 1, 0],		
        [1, 1, 0],		
        [1, 1, 0],		
        [1, 1, 0],		
        [0, 1, 0]],		
        dtype=np.int8)		
    y = np.array([[1],		
        [1],		
        [1],		
        [0],		
        [0],		
        [0],		
        [0],		
        [0],		
        [0],		
        [1]],		
        dtype=np.int8) #* 2 - 1		

    print poisonData(x,y,.3)

# This is the standard boilerplate that calls the main() function.
if __name__ == '__main__':
    main()
