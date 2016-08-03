# coding: utf-8

import numpy as np
import pandas as pd
#from sklearn.linear_model import SGDClassifier as weak_learner
from sklearn.tree import DecisionTreeClassifier as weak_learner
from sklearn.ensemble import AdaBoostClassifier
from metrics import computeError

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

def boost(X, y, sample_weights, learning_rate=.1, epochs=100, terminate=False):
    N = X.shape[0]

    classifier = weak_learner(max_depth=5)
    
    # sklearn implementation requires y to be flattened
    y = np.ravel(y)
    sample_weights = sample_weights/np.sum(sample_weights)    
        
    # 1. Train weak learner with sample_weights distrubtion    
    classifier.fit(X, y, sample_weight=sample_weights)
    
    predictions = classifier.predict(X)
    print 'Predictions:', predictions    
    
    indicator = (predictions != y).astype(int)
    print 'Indicator:', indicator
    
    # 2. Measure the error based on the probability distribution
    error = np.dot(sample_weights, indicator)
    
    # 3. If error == 0, we fit the data perfectly
    if error == 0:
        terminate = True
        alpha = float('inf')
        return classifier, sample_weights, alpha, terminate
    
    # 4. Determine the weight of classifier
    alpha = .5 * np.log((1-error)/error)
    print 'e=%.5f a=%.5f'%(error, alpha)
    
    # 5. Update the distribution
    new_sample_weights = sample_weights * np.exp(-alpha*indicator)
    print 'New weights:', new_sample_weights
    
    return classifier, new_sample_weights, alpha, terminate

def fit(X, y, learning_rate=.1, epochs=100, no_predictors=50, add_bias=True):
    N = X.shape[0]
    classifiers = []
    
    # 0. Initialize uniform weights
    sample_weights = np.array([float(1)/N,] * N)
    print 'Initial Weights:', sample_weights
    
    for iter in xrange(no_predictors):
        classifier, sample_weights, alpha, terminate = boost(X, y, sample_weights)
        classifiers.append(classifier)
                
        if terminate:
            return classifier
    
    return classifiers
    
    # FIX TO INCLUDE EACH ESTIMATOR WEIGHT
def predict(X, classifiers):
    predictions = np.zeros(X.shape[0])

    for classifier in classifiers:
        prediction = classifier.predict
        prediction[prediction==0] = -1
        predictions += prediction
        
    
    
def main():

    
    df_x = pd.read_csv('../Datasets/TrainData/X_train_0.csv', header = None)
    x = np.array(df_x)
    print x
    
    x = addBias(x)
    
    df_y = pd.read_csv('../Datasets/TrainData/y_train_0.csv', header = None)
    y = np.array(df_y)
    print y   
    
    
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
    '''
    w = fit(x,y)
    print 'Final Classifier:', w
    
    predictions = w.predict(x)
    print 'Final Predictions:', predictions
    
    error = computeError(np.ravel(y), predictions)
    print 'Final Error:', error
    
    classifier = AdaBoostClassifier(n_estimators=50, learning_rate=.1)
    classifier.fit(x,np.ravel(y))
    pred = classifier.predict(x)
    print pred
    err = computeError(np.ravel(y), pred)
    print 'SKLEARN ERROR:', err
    
    return

if __name__ == '__main__':
    main()