# coding: utf-8

import numpy as np

def labelSwitching(y, perc_label_switching):
    '''
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

def main():
    '''Test Label Switching'''
    ## dummy data
    ## 10 training samples and 3 features
    ## so a 10 * 3 matrix 
    
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
                  
    print y
    
    print labelSwitching(y, .1)
        
    
# This is the standard boilerplate that calls the main() function.
if __name__ == '__main__':
    main()