# coding: utf-8

import numpy as np

def labelSwitching(y, perc_label_switching):
    '''
    Returns training labels with some percentage labels switched from 1 to 0
    (and vice versa).
    
    Inputs:
    - y: N * 1 Numpy vector of binary values (0 and 1)
         with N: the number of label instances
    - perc_label_switching: real number between 0 and 1; percentage of labels
                            to switch
    
    Output:
    - y: N * 1 Numpy label switched vector of binary values (0 and 1)
    '''
    no_instances = y.shape[0]
    
    no_labels_switched = int(round(no_instances * perc_label_switching))
    
    select_labels = np.sort(np.random.choice(no_instances, no_labels_switched, replace=False))
    
    y[select_labels] = 1 - y[select_labels]
    
    return y
    