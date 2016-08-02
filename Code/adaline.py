# coding: utf-8

'''
Adapted from: https://github.com/galvanic/adversarialML/blob/master/adaline.py

Implementation of the Adaline model.
Training is done using batch gradient descent.
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
    Calculate cost using means squared
    
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
        initial_weights=None,
        learning_rate=0.1,
        termination_condition=None,
        threshold=1e-5,
        ham_label=0,
        spam_label=1,
        verbose=True,
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
    - initial_weights: D * 1 Numpy vector, beginning weights
    - learning_rate: learning rate, a float between 0 and 1
    - termination_condition: returns a bool
    
    Output:
    - W: D * 1 Numpy vector of real values
    '''   
    if not termination_condition:
        termination_condition = max_iters(100)
        
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
    epoch = 1
    last_epoch_error = 1e6

    while not termination_condition():
        ## current iteration classifier output
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
        
        #current_cost = computeCost(Y, O)
        #cost.append(current_cost)
        
        # THINK ABOUT THE EPOCH VALUE
        if (np.abs(last_epoch_error - current_error) < threshold):
            break
            
        last_epoch_error = current_error

        #if verbose and (epoch%10 == 0): print('iteration %d:\tcost = %.3f' % (epoch, cost[-1]))
        epoch += 1

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


def main():
    '''Test Adaline training'''

    df_x = pd.read_csv('../Datasets/TrainData/X_train_0.csv', header = None)
    x = np.array(df_x)
    
    df_y = pd.read_csv('../Datasets/TrainData/y_train_0.csv', header = None)
    y = np.array(df_y)   
   
    df_x_test = pd.read_csv('../Datasets/TestData/X_test_0.csv', header = None)
    x_test = np.array(df_x_test)
    
    df_y_test = pd.read_csv('../Datasets/TestData/y_test_0.csv', header = None)
    y_test = np.array(df_y_test)
   
    ## train model
    weights = fit(features=x, labels=y)
        

    predictions = predict(x_test, weights)
    
    print computeError(y_test, predictions)
    
    return

if __name__ == '__main__':
    sys.exit(main())
