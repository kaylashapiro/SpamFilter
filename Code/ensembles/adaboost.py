# coding: utf-8

import sys
import numpy as np
from sklearn.tree import DecisionTreeClassifier as weak_learner

sys.path.insert(0, '../helpers')

import metrics as met

def boost(X, y, sample_weight, learning_rate=.1, epochs=100, epsilon=1e-10):
    '''
    One iteration of the Adaboost algorithm to build a classifier.
    
    Input:
    
    - X: N * D Numpy matrix of binary feature values (0 and 1)
         with N: the number of training examples
         and  D: the number of features for each example
    - y: 1 * N Numpy vector of binary feature values (0 and 1)
    - sample_weight: 1 * N Numpy vector of sample weightings
    - learning_rate: real number step size
    - epochs: number of iterations
    - epsilon: small term to avoid divide by zero error
    
    Output:
    - classifier: trained classifier object
    - new_sample_weight: 1 * N Numpy vector of sample weightings
    - alpha: weight of the classifier prediction
    '''
    N = X.shape[0]

    classifier = weak_learner(max_depth=1)
    
    # sklearn implementation requires y to be flattened
    y = np.ravel(y)
    sample_weights = sample_weight/np.sum(sample_weight)    
        
    # 1. Train weak learner with sample_weights distrubtion    
    classifier.fit(X, y, sample_weight=sample_weight)
    
    predictions = classifier.predict(X)
    
    indicator = (predictions != y).astype(int)
    
    # 2. Measure the error based on the probability distribution
    error = np.dot(sample_weight, indicator)
    
    # 3. Determine the weight of classifier
    alpha = .5 * np.log((1-error + epsilon)/error + epsilon)
    
    # 4. Update the distribution
    new_sample_weight = sample_weight * np.exp(-alpha*indicator)
    
    return classifier, new_sample_weight, alpha

def fit(X, y, 
        learning_rate=.1, 
        epochs=100, 
        no_predictors=50, 
        ham_label=0,
        spam_label=1):
    '''
    AdaBoost classifier training.
    
    One iteration of the Adaboost algorithm to build a classifier.
    
    Input: 
    - X: N * D Numpy matrix of binary feature values (0 and 1)
         with N: the number of training examples
         and  D: the number of features for each example
    - y: 1 * N Numpy vector of binary feature values (0 and 1)
    - learning_rate: real number step size
    - epochs: number of iterations
    - no_predictors: number of iterations of boosting
    
    /!\ Assumes bias term is already in the X input.
    
    Output:
    - classifiers: list of classifier objects at each boosting 
                   iteration
    - alphas: weights of each classifier's prediction
    '''
    N = X.shape[0]
    classifiers = []
    alphas = []
    
    # 0. Initialize uniform weights
    sample_weight = np.array([float(1)/N,] * N)
    
    for iter in xrange(no_predictors):
        classifier, sample_weight, alpha = boost(X, y, sample_weight)
        classifiers.append(classifier)
        alphas.append(alpha)
                           
    return (classifiers, alphas)
    
def predict(X, classifiers, alphas):
    '''
    Adaboost prediction.
    
    Input:
    - X: N * D Numpy matrix of binary feature values (0 and 1)
         with N: the number of test examples
         and  D: the number of features for each example    
    - classifiers: list of classifier objects at each boosting 
                   iteration
    - alphas: weights of each classifier's prediction
    
    /!\ Assumes bias term is already in the X input.
    
    Output:
    - y_predictions: 1 * N Numpy vector of binary values (0 and 1);
                     predicted classes
    '''
    predictions = np.zeros(X.shape[0])
    no_classifiers = len(classifiers)
    
    for iter in xrange(no_classifiers):
        prediction = classifiers[iter].predict(X)
        prediction[prediction==0] = -1
        predictions += alphas[iter]*prediction
    
    y_predictions = np.sign(predictions)
    y_predictions[y_predictions==-1] = 0
    
    return y_predictions
   