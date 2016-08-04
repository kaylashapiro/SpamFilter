# coding: utf-8

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_curve, auc
from metrics import computeError, computeMetrics, computeROC, computeRates
from helpers import addBias
import matplotlib.pyplot as plt

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

def computeCost(trueValues, predictions):
    '''
    Returns the average cost associated with each prediction for an
    example given its true value.
    
    Inputs:
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
    
def gradient_update(theta, x, y, alpha):
    '''
    Returns the gradient update for a single training example.
    
    Inputs:
    - theta: D * 1 Numpy vector of float weight values
    - x: 1 * D Numpy vector of binary feature values (0 and 1)
         representing one training example
         with D: the number of features for each example
                 (including bias term)
    - y: 1 * 1 Numpy vector of binary feature values (0 and 1)
         representing the class label of training example 'x'
    
    Ouput:
    - new_theta: D * 1 Numpy vector of float updated weight values
    '''

    s = sigmoid(np.dot(x, theta))
    grad = np.multiply((s - y), x)
    new_theta = theta - np.reshape(np.dot(alpha, grad),(theta.shape[0],1))
    
    return new_theta  
   
def fit(X, y, alpha=.1, epochs=100, threshold=1e-5, add_bias = True):
    '''
    Implementation of logistic regression classifier with stochastic
    gradient descent.
    
    Inputs:
    - X: N * D Numpy matrix of binary feature values (0 and 1)
         with N: the number of training examples
         and  D: the number of features for each example
    - y: N * 1 Numpy vector of binary feature values (0 and 1)
    - alpha: float; stepsize to take along the gradient
    - epochs: integer; number of full passes through the training set
    - threshold: float termination condition; if error produced in
                 an epoch is within the threshold of the error produced
                 in the previous epoch, then break
    
    Output:
    - theta: D * 1 Numpy vector of float weights of trained classifier
    '''
    
    if (add_bias):    
        X = addBias(X)
    
    N, D = X.shape
    theta = np.zeros((D,1))  
    epoch_errors = np.zeros((epochs,1))
    last_epoch_error = 1e6
 
    perm = np.random.permutation(N) 
    
    for i in range(epochs):
        X = X[perm,:]
        y = y[perm,:]
        
        for j in range(N):
            theta = gradient_update(theta, X[j,:], y[j], alpha)   
            
        epoch_errors[i,:] = computeError(y, predict(X, theta, False))
        
        if (np.abs(last_epoch_error - epoch_errors[i]) < threshold):
            break
            
        last_epoch_error = epoch_errors[i] 
        
    #plt.figure()
    #plt.plot(epoch_errors)
    #plt.show()
    
    return theta
    
def predict(X, theta, add_bias=True):
    '''
    Inputs:
    - X: N * D Numpy matrix of binary feature values (0 and 1)
         with N: the number of training examples
         and  D: the number of features for each example
    - theta: D * 1 Numpy vector of current float weights of classifier
    - add_bias: default boolean
    
    Output:
    - predictions: N * 1 Numpy vector of binary values (0 and 1);
                   predicted classes
    '''
    
    if (add_bias):    
        X = addBias(X)
        
    probs = sigmoid(np.dot(X, theta))
    
    predictions = np.zeros((X.shape[0],1))   
    predictions[probs>0.5] = 1
    
    return predictions

def predictSoft(X, theta, add_bias=True):
    '''
    Inputs:
    - X: N * D Numpy matrix of binary feature values (0 and 1)
         with N: the number of training examples
         and  D: the number of features for each example
    - theta: D * 1 Numpy vector of current float weights of classifier
    - add_bias: default boolean
    
    Output:
    - predictions: N * 1 Numpy vector of float values between
                   0 and 1 representing the probability each 
                   training example is 1
    '''
    
    if (add_bias):    
        X = addBias(X)
        
    probs = sigmoid(np.dot(X, theta))
    
    return probs 
    
    
def main():

    df_x = pd.read_csv('../Datasets/TrainData/X_train_0.csv', header = None)
    x = np.array(df_x)
    print x
    
    df_y = pd.read_csv('../Datasets/TrainData/y_train_0.csv', header = None)
    y = np.array(df_y)
    print y   
    
    w = fit(x,y,0.1)
    
    pred = predict(x,w)
    
    print computeError(y,pred)
    
    return
    

if __name__ == '__main__':
    main()

    
