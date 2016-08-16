# coding: utf-8

'''
Implementation of a simple dictionary attack. 

Injects copies of same malicious data point.

Assumes no bias has been added yet.
'''

import numpy as np
import random

def vary_attack_points(no_poisoned, D, d, threshold):
    '''
    Returns crafted attack instances. Generates varying attack points based
    on a fraction of the attacker's full feature knowledge.
    
    Input:
    - no_poisoned: number of poison data examples
    - D: the number of features for each example
    - d: the number of features the attacker knows about
    - threshold: float between 0 and 1
                 fraction of known features the attacker uses in point generation; randomness
    
    Output:
    - attack_points: no_poisoned * D Numpy matrix of poison examples
    '''
    ## Attacker can choose whether or not to use their feature knowledge
    no_poisoned_features = int(round(threshold * d))
    
    ## Generate no_poisoned different attack instances
    rand_features = np.array([0] * (D - no_poisoned_features) + [1] * no_poisoned_features)
    np.random.shuffle(rand_features)
    
    attack_points = [rand_features]
    
    for instance in xrange(1, no_poisoned):
        np.random.shuffle(rand_features)
        attack_points = np.vstack((attack_points, [rand_features]))
        
    return attack_points
    

def simple(no_poisoned, D, d, **kwargs):
    '''
    Returns crafted attack instances. Takes advantage of the attacker's
    full feature knowledge and generates copies of the same attack point.
    
    Input:
    - no_poisoned: number of poison data examples
    - D: the number of features for each example
    - d: the number of features the attacker knows about
    
    Output: 
    - attack_points: no_poisoned * D Numpy matrix of poison examples
    '''
    ## Generate attack point mimicking attacker knowledge
    rand_features = np.array([0] * (D - d) + [1] * d)
    np.random.shuffle(rand_features)
    
    attack_points = np.array([rand_features,] * no_poisoned)
    
    return attack_points
    

def poisonData(features, labels, 
               ## params
               percentage_samples_poisoned,
               percentage_features_poisoned=1.0,
               generate_attack_data=simple,
               threshold=1.0,
               ham_label=0,
               ):
    '''
    Returns the input data with *added* data that is crafted specifically to cause
    a poisoning dictionary attack, where all features (within attacker knowledge)
    are set to 1. 
    
    Input:
    - features: N * D Numpy matrix of binary feature values (0 and 1)
                with N: the number of training examples
                and  D: the number of features for each example
    - labels: N * 1 Numpy vector of binary values (0 and 1)
    - percentage_samples_poisoned: float between 0 and 1 
                                   fraction of the dataset under the attacker's control
    - percentage_features_poisoned: float between 0 and 1
                                    fraction of knowledge the attacker has of the feature set
    - threshold: float between 0 and 1
                 fraction of known features the attacker uses in point generation; randomness
    
    Output:
    - X: poisoned features
    - Y: poisoned labels    
    '''
    ## notation
    X, Y = features, labels
    N, D = X.shape
    
    no_poisoned = int(round(percentage_samples_poisoned * N))
    d = int(round(percentage_features_poisoned * D))
        
    attack_points = generate_attack_data(no_poisoned, D, d, threshold)
    
    poisoned_indices = np.random.choice(N, no_poisoned, replace=False)
        
    X[poisoned_indices] = attack_points
    Y[poisoned_indices] = 1 # Contamination Assumption
    
    return (X, Y)
    