# coding: utf-8

'''
Adapted from: https://github.com/galvanic/adversarialML/blob/master/empty.py

Implementation of an empty attack.

Assumes no bias has been added yet.
'''
import numpy as np

def poisonData(features, labels,
        ## params
        percentage_samples_poisoned, 
        percentage_features_poisoned=1.0,
        ham_label=0,
        spam_label=1,
        ):
    '''
    Returns the input data with *replaced* data that is crafted specifically to
    cause a poisoning empty attack, where all features are set to zero.
    Inputs:
    
    - features: N * D Numpy matrix of binary values (0 and 1)
                with N: the number of training examples
                and  D: the number of features for each example
    - labels: N * 1 Numpy vector of binary values (-1 and 1)
    - percentage_samples_poisoned: float between 0 and 1
        percentage of the dataset under the attacker's control
        
    Outputs:
    - X: poisoned features
    - Y: poisoned labels
    '''
    ## notations
    X, Y = features, labels
    N, D = X.shape ## number of N: samples, D: features
    
    no_poisoned = int(round(N * percentage_samples_poisoned))

    ## randomly replace some samples with the poisoned ones
    ## so that total number of samples doesn't change
    poisoned_indices = np.random.choice(N, no_poisoned)
    X[poisoned_indices] = 0

    ## the contamination assumption
    Y[poisoned_indices] = spam_label

    return (X, Y)