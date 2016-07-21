# coding: utf-8

'''
Adapted from: https://github.com/galvanic/adversarialML/blob/master/adaline.py
Implementation of the Adaline model.
Training is done using batch gradient descent.

TODO ? make an Adaline class with train and test as methods
TODO ? implement regularisation
TODO ? cost and error could be measured outside the function
     or at least use a callable to calculate them, otherwise duplicated code
     across models
TODO clean up the code further, especially duplicated sections (adaline model
     etc.)
'''
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from metrics import computeError
from gradientDescent import max_iters

def addBias(X): 
    '''
    Adds bias term to the training set.
    
    Input:
    
    - X: N * D Numpy matrix of binary feature values (0 and 1)
         with N: the number of training examples
         and  D: the number of features for each example
    
    Output:
    - X_bias: N * (D + 1) Numpy matrix of binary feature values
              consisting of a column of ones + X
    '''
    
    X_bias = np.insert(X, 0, 1, axis=1)
    
    return X_bias
    

def computeCost(trueValues, predictions):
    '''
    Calculate cost using Means Squared
    '''
    trueValues, predictions = map(np.ravel, [trueValues, predictions]) ## make sure shape is (len,) for both
    cost = np.mean(np.square(trueValues - predictions))
    
    return cost
    

def fit(features, labels,
        ## params:
        initial_weights=None,
        learning_rate=0.01,
        termination_condition=max_iters(100),
        ham_label=0,
        spam_label=1,
        verbose=False,
        ):
    '''
    Returns the optimal weights for a given training set (features
    and corresponding label inputs) for the ADALINE model.
    These weights are found using the gradient descent method.
    
    !!!! ASSUMES BIAS HAS BEEN ADDED !!!!!
    
    TRAINING PHASE
    Inputs:
    - features: N * D Numpy matrix of binary values (0 and 1)
        with N: the number of training examples
        and  D: the number of features for each example
    - labels:   N * 1 Numpy vector of binary values (-1 and 1)
    - initial_weights: D * 1 Numpy vector, beginning weights
    - learning_rate: learning rate, a float between 0 and 1
    - termination_condition: returns a bool
    Output:
    - W: D * 1 Numpy vector of real values
    TODO yield cost, error, weights as it is learning ?
         this could allow possibility to inject new learning rate during
         training
    TODO implement an autostop if cost is rising instead of falling ?
    '''
    ## 0. Prepare notations
    
    X, Y = features, labels
    N, D = features.shape   # N #training samples; D #features
    cost = []               # keep track of cost
    error = []              # keep track of error

    ## 1. Initialise weights
    W = np.zeros((D, 1)) if initial_weights is None else initial_weights.reshape((D, 1))

    ## 2. Evaluate the termination condition
    i = 1
    while not termination_condition():

        ## current iteration classifier output
        O = np.dot(X, W)

        ## specialty of ADALINE is that training is done on the weighted sum,
        ## _before_ the activation function
        ## batch gradient descent
        gradient = -np.mean(np.multiply((Y - O), X), axis=0)
        gradient = gradient.reshape(W.shape)

        ## 3. Update weights
        W = W - learning_rate * gradient

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

        if verbose: print('iteration %d:\tcost = %.3f' % (i, cost[-1]))
        i += 1

    return W, cost, error


def predict(parameters, features,
        ## params
        ham_label=0,
        spam_label=1,
        ):
    '''
    TEST PHASE
    TODO not sure what makes sense to measure here ?
         => performance can be calculated outside the function
    '''
    ## notation
    X, W = features, parameters
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


def main():
    '''Test Adaline training'''
    ## dummy data
    ## 10 training samples and 3 features
    ## so a 10 * 3 matrix 
    '''
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
    '''
    df_x = pd.read_csv('Features.csv', header = None)
    x = np.array(df_x)
    x = addBias(x)
    print x
    
    df_y = pd.read_csv('Labels.csv', header = None)
    y = np.array(df_y)
    print y   
                    
    ## train model
    weights, cost, error = fit(features=x, labels=y,
        learning_rate=1,
        termination_condition=max_iters(100))
    print('weights: %s' % ([' %.3f' % w for w in weights]))
    #print cost
    #print error
    predictions = predict(weights, x)
    print predictions
    
    plt.figure(figsize=(10,6), dpi=120)
        
    plt.plot(error, color='blue', label='Error')
    plt.plot(cost, color='red', label = 'Cost')
    
    #plt.title('')
    plt.xlabel('Iterations')
    plt.ylabel('Rate')
    
    #plt.xlim(0, no_predictors + 1)
    #plt.ylim(0)
        
    plt.legend(loc='upper right', frameon=True)
    
    plt.show()
    
    return

if __name__ == '__main__':
    sys.exit(main())
