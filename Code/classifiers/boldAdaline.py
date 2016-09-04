# coding: utf-8

'''
Adapted from: https://github.com/galvanic/adversarialML/blob/master/adaline.py

Implementation of the Adaline model with 'bold driver' adaptive learning rate.

Training is done using batch gradient descent.
'''
import sys
import numpy as np

sys.path.insert(0, '../helpers')

from add_bias import addBias
from metrics import computeError

def computeCost(trueValues, predictions):
    '''
    Calculate cost using means squared
    
    Input:
    - trueValues: N * 1 Numpy vector of binary labels
                  with N: the number of examples
    - predictions: N * 1 Numpy vector of predicted values
    
    Output:
    - cost: real number calculated using means squared.
    '''  
    trueValues, predictions = map(np.ravel, [trueValues, predictions]) ## make sure shape is (len,) for both
    
    cost = np.mean(np.square(trueValues - predictions))
    
    return cost
    

def fit(features, labels,
        ## params:
        initial_weights=None,
        learning_rate=0.1,
        max_epoch = 200,
        threshold=1e-5,
        ham_label=0,
        spam_label=1,
        add_bias = True
        ):
    '''
    Returns the optimal weights for a given training set (features
    and corresponding label inputs) for the ADALINE model.
    These weights are found using the gradient descent method.
        
    Inputs:
    - features: N * D Numpy matrix of binary values (0 and 1)
        with N: the number of training examples
        and  D: the number of features for each example
    - labels:   N * 1 Numpy vector of binary values (-1 and 1)
    - learning_rate: float between 0 and 1
    - initial_weights: D * 1 Numpy vector, beginning weights
    
    Output:
    - W: D * 1 Numpy vector of real values
    '''           
    if (add_bias):
        features = addBias(features)
    
    ## 0. Prepare notations
    X, Y = features, labels
    N, D = features.shape   # N #training samples; D #features
    cost = []               # keep track of cost
    error = []              # keep track of error

    ## 1. Initialise weights
    W = np.zeros((D, 1)) if initial_weights is None else initial_weights.reshape((D, 1))

    ## 2. Evaluate the termination condition
    epoch = 0
    last_epoch_error = 1e10

    while epoch < max_epoch:
        ## current iteration classifier output
        
        last_W = W
        
        O = np.dot(X, W)

        ## specialty of ADALINE is that training is done on the weighted sum,
        ## _before_ the activation function
        ## batch gradient descent
        gradient = -np.mean(np.multiply((Y - O), X), axis=0)

        ## 3. Update weights
        W = W - learning_rate * gradient.reshape(W.shape)

        ## Keep track of error and cost (weights from previous iteration)
        ## T is equivalent to threshold/step activation function
        if ham_label is 0:               ## spam label assumed 1
            T = np.zeros(O.shape)
            T[O > 0.5] = 1
        else:   ## ham label is assumed -1, spam label assumed 1
            T = np.ones(O.shape)
            T[O < 0] = -1

        current_error = computeError(T, Y)
        error.append(current_error)
        
        current_cost = computeCost(Y, O)
        cost.append(current_cost)
        
        if (current_error < last_epoch_error): 
            current_error = computeError(T, Y)
            error.append(current_error)
            learning_rate = learning_rate*1.05
            #print 'LEARNING RATE INCREASED TO:', learning_rate
        elif (current_error - last_epoch_error > 1e-8):
            learning_rate = learning_rate*.5
            W = last_W
            epoch -= 1
            current_error = 1e10
            #print 'LEARNING RATE DECREASED TO:', learning_rate
            #print 'UNDO ITERATION!!!!!!'
        else:
            current_error = computeError(T, Y)
            error.append(current_error)
            
        #if (epoch > 50 and np.abs(last_epoch_error - current_error) < threshold):
        #    break
            
        epoch += 1
            
        last_epoch_error = current_error

        #if verbose and (epoch%1 == 0): print('iteration %d:\tcost = %.3f \terror = %.3f' % (epoch, cost[-1], error[-1]))
       
    return W


def predict(features, weights,
        ## params
        ham_label=0,
        spam_label=1,
        add_bias = True
        ):
    '''
    Input:
    - features: N * D Numpy matrix of binary values (0 and 1)
        with N: the number of training examples
        and  D: the number of features for each example
    - weights: D * 1 Numpy vector of real values
    
    Output:
    - T: N * 1 Numpy vector of binary prediction values
    '''
    if (add_bias):
        features = addBias(features)
    
    ## notation
    X, W = features, weights
    N, D = features.shape
    
    ## apply model
    O = np.dot(X, W)

    ## calculate output
    ## T is equivalent to threshold/step activation function
    if ham_label is 0:               ## spam label assumed 1
        T = np.zeros(O.shape)
        T[O > 0.5] = 1
    else:   ## ham label is assumed -1, spam label assumed 1
        T = np.ones(O.shape)
        T[O < 0] = -1
    
    return T
