# coding: utf-8

'''
Adapted from: https://github.com/galvanic/adversarialML/blob/master/helpers/gradientdescent.py
'''

import numpy as np
from collections import deque
from math import ceil
from metrics import computeError
    
def adagrad(features, labels,
            ## functions specific to classifier:
            calculate_output,
            cost_function,
            predict,
            ## params:
            batch_size=1,
            learning_rate=.1,
            max_epochs=100,
            initial_weights=None,
            convergence_threshold=1e-5,
            convergence_look_back=1,
            smoothing_term=1e-8,
            ):
    '''
    Returns the optimal weights for a given training set and a given model 
    using the gradient descent method. The model is determined by the 
    'calculate_output', 'cost_function' and 'predict' functions.
    
    /!\ Assumes bias term is already in the features input.
    
    Input:
    - features: N * D Numpy matrix of binary values (0 and 1)
                with N: the number of training examples
                and  D: the number of features for each example
    - labels: N * 1 Numpy vector of binary values (0 and 1)
    - batch_size: int between 1 and N
                    1 = stochastic gradient descent
                    N = batch gradient descent
                    everything in between = mini-batch gradient descent
    - learning_rate: float, between 0 and 1
    - max_epochs: int, >= 0; maximum number of times to run through training set
    - initial_weights: D * 1 Numpy vector of feature weights
    - convergence_threshold: float, very small number; e.g. 1e-5
    - convergence_look_back: int, >= 1
                             stops if the error difference hasn't been over threshold
                             for the last X epochs.
    
    Output:
    - W: D * 1 Numpy vector of real values
    
    TODO adaptive learning rate
    '''    
    ## notation
    X, Y = features, labels
    N, D = X.shape # N training samples; D features
    
    ## initialize weights
    W = np.zeros((D,1)) if initial_weights is None else initial_weights.reshape((D, 1))
    
    ## evaluate the termination conditions
    previous_errors = deque(maxlen=convergence_look_back)
    previous_errors.append(1e6)
    
    ## store gradient sum of squares
    gti = np.zeros(D)
    
    epoch = 0
    while epoch < max_epochs:
        ## mix up samples (they will therefore be fed in different order
        ## at each training) -> commonly accepted to improve gradient
        ## descent, making convergence faster
        permuted_indices = np.random.permutation(N)
        
        X = X[permuted_indices, :]
        Y = Y[permuted_indices]
        #print X.shape
        #print Y.shape
        
        for instance in xrange(N):
            x = X[instance]
            y = Y[instance]
        
            ## classifier output of current training instance
            o = calculate_output(x, W)
            
            ## gradient descent: minimize the cost function
            ## gradient equation was obtained by deriving the LMS cost function
            gradient = -np.multiply((y - o), x)
            
            gti += gradient ** 2
            
            adjusted_gradient = gradient / (np.sqrt(gti) + smoothing_term) 

            ## update weights
            W = W - learning_rate * adjusted_gradient.reshape(W.shape)
            
        ## Keep track of cost and error
        P = predict(X, W)
        error = computeError(Y, P)
        cost = cost_function(Y, P)
         
        previous_errors.append(error) 
         
        ## check for convergence in last x epochs
        if all(abs(np.array(previous_errors) - error) < convergence_threshold):
            return W
        epoch += 1
        
    return W
