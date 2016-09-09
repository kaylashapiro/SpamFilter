# coding: utf-8

'''
Adapted from: https://github.com/galvanic/adversarialML/blob/master/adaline.py

Implementation of the Adaline model.

Training is done using batch gradient descent.
'''
import sys
import numpy as np

sys.path.insert(0, '../helpers')

from gradientdescent import gradient_descent 

def calculate_output(X, W):
    return np.dot(X, W)

def computeCost(trueValues, predictions):
    '''
    Calculate cost using means squared.
    
    Input:
    - trueValues: N * 1 Numpy vector of binary labels
                  with N: the number of examples
    - predictions: N * 1 Numpy vector of predicted values
    
    Output:
    - cost: real number calculated using means squared
    '''  
    trueValues, predictions = map(np.ravel, [trueValues, predictions]) ## make sure shape is (len,) for both
    
    cost = np.mean(np.square(trueValues - predictions))
    
    return cost
    
def fit(features, labels,
        ## params:
        batch_size=1,
        max_epochs=100,
        learning_rate=.1,
        initial_weights=None,
        convergence_threshold=1e-5,
        convergence_look_back=2,
        ham_label=0,
        spam_label=1,
        ):
    '''
    Returns the optimal weights for a given training set (features
    and corresponding label inputs) for the ADALINE model.
    These weights are found using the gradient descent method.
    
    /!\ Assumes bias term is already in the features input.
        
    Inputs:
    - features: N * D Numpy matrix of binary values (0 and 1)
                with N: the number of training examples
                and  D: the number of features for each example
    - labels:   N * 1 Numpy vector of binary values (0 and 1)
    - learning_rate: float between 0 and 1
    - initial_weights: D * 1 Numpy vector, beginning weights
    - convergence_threshold: float, very small number; e.g. 1e-5
    - convergence_look_back: int, >= 1
                             stops if the error difference hasn't been over threshold
                             for the last X epochs
        
    Output:
    - W: D * 1 Numpy vector of real values
    '''           
    ## notation
    X, Y = features, labels
    N, D = X.shape   # N #training samples; D #features
    
    W = gradient_descent(X, Y,
                         calculate_output,
                         computeCost,
                         predict,
                         batch_size=N,
                         learning_rate=learning_rate,
                         max_epochs=max_epochs,
                         initial_weights=initial_weights,
                         convergence_threshold=convergence_threshold,
                         convergence_look_back=convergence_look_back,
                         )

    return W


def predict(features, parameters,
        ## params
        ham_label=0,
        spam_label=1,
        ):
    '''
    Input:
    - features: N * D Numpy matrix of binary values (0 and 1)
                with N: the number of training examples
                and  D: the number of features for each example
    - parameters: output of fit method
                  D * 1 Numpy vector of current float weights of classifier
    
    /!\ Assumes bias term is already in the features input.
    
    Output:
    - T: N * 1 Numpy vector of binary prediction values (0 and 1)
    '''
    
    ## notation
    X, W = features, parameters
    N, D = features.shape
    
    ## apply model
    O = calculate_output(X, W)

    ## calculate output
    ## T is equivalent to threshold/step activation function
    if ham_label is 0:               ## spam label assumed 1
        T = np.zeros(O.shape)
        T[O > 0.5] = 1
    else:   ## ham label is assumed -1, spam label assumed 1
        T = np.ones(O.shape)
        T[O < 0] = -1
    
    return T
