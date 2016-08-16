# coding: utf-8

'''
Adapted from: https://github.com/galvanic/adversarialML/blob/master/helpers/gradientdescent.py
'''

import numpy as np
import sys
from collections import deque
from math import ceil
from metrics import computeError

def get_batch(X, Y, permuted_indices, batch_number, batch_size):
    '''
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
    print x
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
        dtype=np.int8) 
    print y

    N = x.shape[0]
    permuted_indices = np.random.permutation(N)
    print permuted_indices
    print get_batch(x, y, permuted_indices, 3, 3)

# This is the standard boilerplate that calls the main() function.
if __name__ == '__main__':
    main()
