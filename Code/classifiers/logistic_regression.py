# coding: utf-8

'''
Implementation of the logistic regression model.

Training is done using stochastic gradient descent.
'''

import sys
import numpy as np

sys.path.insert(0, '../helpers')

from gradientdescent import gradient_descent

def sigmoid(z):
    '''
    Definition of the sigmoid function. Returns a float between
    0 and 1.
    
    Input:
    - z: float
    
    Output:
    - prediction: float between 0 and 1
    '''
    
    prediction = np.divide(1, 1 + np.exp(-z))

    return prediction
    
def calculate_output(X, W):
    return sigmoid(np.dot(X,W))

def computeCost(trueValues, predictions):
    '''
    Returns the average cost associated with each prediction for an
    example given its true value.
    
    Input:
    - trueValues:  N * 1 Numpy vector of binary values (0 and 1)
                   with N: the number of training examples
    - predictions: N * 1 Numpy vector of float values between 0 
                   and 1; each value represents the probability
                   that the class for a given example is 1.
                   
    Output:
    - cost: float; average cost over a set of training examples
    '''
    
    costs = - np.multiply(trueValues, np.log(predictions)) - np.multiply((1 - trueValues),(1 - np.log(predictions)))
    
    cost = np.mean(costs)
    
    return cost 
   
def fit(features, labels,
        ## params
        batch_size=1,
        max_epochs=100,
        learning_rate=.1,
        initial_weights=None,
        convergence_threshold=1e-5,
        convergence_look_back=1,
        ham_label=0,
        spam_label=1,
        ):
    '''
    Implementation of logistic regression classifier with stochastic
    gradient descent.
    
    /!\ Assumes bias term is already in the features input.
    
    Input:
    - features: N * D Numpy matrix of binary feature values (0 and 1)
                with N: the number of training examples
                and  D: the number of features for each example
    - labels: N * 1 Numpy vector of binary feature values (0 and 1)
    - learning_rate: float between 0 and 1
    - initial_weights: D * 1 Numpy vector, beginning weights
    
    Output:
    - W: D * 1 Numpy vector of float weights of trained classifier
    '''  
    ## notation
    X, Y = features, labels
    N, D = X.shape   # N #training samples; D #features
    
    W = gradient_descent(X, Y,
                         calculate_output,
                         computeCost,
                         predict,
                         batch_size=batch_size,
                         learning_rate = learning_rate,
                         max_epochs=max_epochs,
                         initial_weights=initial_weights,
                         convergence_threshold=convergence_threshold,
                         convergence_look_back=convergence_look_back,
                         )
    
    return W
    
def predict(features, parameters):
    '''
    Input:
    - features: N * D Numpy matrix of binary feature values (0 and 1)
                with N: the number of examples
                and  D: the number of features for each example
    - parameters: output of fit method
                  D * 1 Numpy vector of current float weights of classifier
    
    /!\ Assumes bias term is already in the X input.
    
    Output:
    - predictions: N * 1 Numpy vector of binary values (0 and 1);
                   predicted classes
    '''
    probs = calculate_output(features, parameters)
    
    predictions = np.zeros((features.shape[0],1))   
    predictions[probs>0.5] = 1
    
    return predictions   
