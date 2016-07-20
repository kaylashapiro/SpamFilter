# coding: utf-8

'''
Adapted from: https://github.com/galvanic/adversarialML/blob/master/empty.py

Implementation of an empty attack.
Assumes no bias has been added yet.
'''
import numpy as np

def poisonData(X, y,
        ## params
        percentage_samples_poisoned,
        ):
    '''
    Returns the input data with *replaced* data that is crafted specifically to
    cause a poisoning empty attack, where all features are set to zero.
    Inputs:
    
    - X: N * D Numpy matrix of binary values (0 and 1)
        with N: the number of training examples
        and  D: the number of features for each example
    - y: N * 1 Numpy vector of binary values (-1 and 1)
    - percentage_samples_poisoned: float between 0 and 1
        percentage of the dataset under the attacker's control
        
    Outputs:
    - X: poisoned features
    - Y: poisoned labels
    '''
    ## notations
    spam_label = 1
    N, D = X.shape ## number of N: samples, D: features
    num_poisoned = int(round(N * percentage_samples_poisoned))

    ## randomly replace some samples with the poisoned ones
    ## so that total number of samples doesn't change
    indices = np.random.choice(N, num_poisoned)
    X[indices] = 0

    ## the contamination assumption
    Y[indices] = spam_label

    return (X, Y)