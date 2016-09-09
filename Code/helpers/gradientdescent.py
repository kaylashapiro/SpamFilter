# coding: utf-8

'''
Adapted from: https://github.com/galvanic/adversarialML/blob/master/helpers/gradientdescent.py
'''

import numpy as np
from collections import deque
from math import ceil
from metrics import computeError

def get_batch(X, Y, permuted_indices, batch_number, batch_size):
    '''
    Input:
    - X: N * D Numpy matrix of binary values (0 and 1
         with N: the number of training examples
         and  D: the number of features for each example
    - Y: N * 1 Numpy vector of binary values (0 and 1)
    - permuted_indices: sample indices mixed
    - batch_number: int
    - batch_size: int, > 0, <= N
    
    Output:
    - x: feature batch
    - y: label batch
    '''
    N = X.shape[0]
    
    if (batch_size == N):
        return (X, Y)
    
    start = (batch_number * batch_size) % N
    end = (batch_number * batch_size + batch_size) % N
    
    end = end if (end and end > start) else None
    
    samples = permuted_indices[start:end]
    
    x, y = X[samples], Y[samples]
    
    return (x, y)
    
def gradient_descent(features, labels,
                     ## functions specific to classifier:
                     calculate_output,
                     cost_function,
                     predict,
                     ## params:
                     batch_size,
                     learning_rate,
                     max_epochs,
                     initial_weights,
                     convergence_threshold,
                     convergence_look_back,
                     adaptive_learning_rate=False
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
    '''    
    ## notation
    X, Y = features, labels
    N, D = X.shape # N training samples; D features
    
    ## initialize weights
    W = np.zeros((D,1)) if initial_weights is None else initial_weights.reshape((D, 1))
    
    ## evaluate the termination conditions
    previous_errors = deque(maxlen=convergence_look_back)
    previous_errors.append(1e6)
    
    epoch = 0
    while epoch < max_epochs:
        ## mix up samples (they will therefore be fed in different order
        ## at each training) -> commonly accepted to improve gradient
        ## descent, making convergence faster
        permuted_indices = np.random.permutation(N)
        
        no_batches = ceil(float(N)/batch_size)
        batch_number = 0
        
        while batch_number < no_batches:
            x, y = get_batch(X, Y, permuted_indices, batch_number, batch_size)
            
            ## classifier output of current batch
            o = calculate_output(x, W)
            
            ## gradient descent: minimize the cost function
            ## gradient equation was obtained by deriving the LMS cost function
            gradient = -np.mean(np.multiply((y - o), x), axis=0)
            
            ## update weights
            W = W - learning_rate * gradient.reshape(W.shape)
            
            batch_number += 1
            
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
