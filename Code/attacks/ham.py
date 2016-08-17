# coding: utf-8

'''
Adapted from: https://github.com/galvanic/adversarialML/blob/master/attacks/ham.py

Implementation of a ham attack.

Inject malicious points indicative of ham email.

Assumes no bias has been added yet.
'''
import numpy as np
import pandas as pd
from sklearn.metrics import mutual_info_score

def select_using_frequency(features, labels, threshold=0, ham_label=0):
    '''
    Returns indices of the most salient features for the ham class,
    using the frequency the features appear in ham instances vs
    spam instances.
    '''
    X, Y = features, np.ravel(labels)
    N, D = X.shape
    
    ##calculate frequency of feature presence relative to each class
    ham_freq  = np.mean(X[np.ravel(Y == ham_label)], axis=0)
    spam_freq = np.mean(X[np.ravel(Y != ham_label)], axis=0)
    
    ##select indicies more frequent in ham
    salient_indices = np.ravel(np.where(ham_freq - spam_freq > threshold))    
    
    return salient_indices

def select_most_present(features, labels, threshold=0, ham_label=0):
    '''
    Returns indices of the most salient features for the ham class, using a
    crude measure of how many times the features appear in ham instances.
    '''
    X, Y = features, labels

    ham_mask = np.ravel(Y == ham_label)
    hams = X[ham_mask]

    ## use features that appear most in ham emails
    count = np.sum(hams, axis=0)
    salient_indices = np.nonzero(count > threshold)[0]

    return salient_indices


def select_using_MI(features, labels, threshold=0.1, ham_label=0):
    '''
    Returns indices of the most salient features for the ham class, using a
    mutual information score between feature values and class label, and
    from the highest scoring, filtering the ones that are most present
    in spam relatively. This makes sense since we then use these indices
    to choose which features to turn on in emails
    TODO or I could keep all the highest MI score features, and if more
         present in ham, I set it to 1, else I set it to zero
         requires extra array somewhere though
         Thinking about it, this is already what is happening since the ham
         malicious instances already have all their features set to zero
         (but this proportion of features controlled by the attacker could
         vary ? depending on attacket's dataset knowledge ? ie. malicious
         instances' feature values could be initialised randomly, or drawn from a
         spammy dicstribution to mimick an email that still has a malicious
         potential (although this isn't necessary since we are doing a poison,
         not evasion attack); then compare this to initialised with 0s or with 1s
    '''
    X, Y = features, np.ravel(labels)
    N, D = X.shape
    d = int(D * threshold) ## percentile of features to keep

    ## calculate frequency of feature presence relative to each class
    ham_freq  = np.mean(X[np.ravel(Y == ham_label)], axis=0)
    spam_freq = np.mean(X[np.ravel(Y != ham_label)], axis=0)

    ## calculate mutual information between features and labels
    MI_per_feature = (mutual_info_score(X[:, f], Y) for f in range(D))
    MI_per_feature = np.fromiter(MI_per_feature, dtype=np.float16)

    ## keep only salient features for ham (according to relative presence in that class)
    MIs = MI_per_feature[ham_freq > spam_freq]
    salient_indices = np.argpartition(MIs, -d)[-d:]
    ## ^ https://stackoverflow.com/questions/10337533/a-fast-way-to-find-the-largest-n-elements-in-an-numpy-array/20177786#20177786

    return salient_indices


def poisonData(features, labels,
               ## params
               percentage_samples_poisoned,
               percentage_features_poisoned=1.0,
               feature_selection_method=select_using_MI,
               threshold=.1,
               ham_label=0,
               spam_label=1,
               ):
    '''
    Returns the input data with *replaced* data that is crafted specifically to
    cause a poisoning ham attack, where features of the contaminating emails
    are selected because they are indicative of the ham class.
    Inputs:
    - features: N * D Numpy matrix of binary values (0 and 1)
        with N: the number of training examples
        and  D: the number of features for each example
    - labels:   N * 1 Numpy vector of binary values (-1 and 1)
    - percentage_samples_poisoned: float between 0 and 1
        percentage of the dataset under the attacker's control
    - percentage_features_poisoned: float between 0 and 1
        percentage of the features under the attacker's control
    - feature_selection_method: string
    - threshold: percentile of features to keep
    Outputs:
    - X: N * D poisoned features
    - Y: N * 1 poisoned labels
    '''
    ## notations
    X, Y = features, labels
    N, D = X.shape ## number of N: samples, D: features
  
    num_poisoned = int(round(N * percentage_samples_poisoned))
    d = int(round(D * percentage_features_poisoned))

    ## find the most salient features, indicative of the ham class
    salient_indices = feature_selection_method(X, Y)
    #print salient_indices

    no_salient_indices = len(salient_indices)
    
    if (no_salient_indices > d):
        salient_indices = np.random.choice(salient_indices, d, replace=False)
        
    ## randomly replace some samples with the poisoned ones
    ## so that total number of samples doesn't change
    poisoned_indices = np.random.choice(N, num_poisoned,replace=False)
    #print poisoned_indices    

    X[poisoned_indices] = 0

    ## "turn on" features whose presence is indicative of ham
    X[np.ix_(poisoned_indices, salient_indices)] = 1

    ## the contamination assumption
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

    print poisonData(x,y,.1)

# This is the standard boilerplate that calls the main() function.
if __name__ == '__main__':
    main()
