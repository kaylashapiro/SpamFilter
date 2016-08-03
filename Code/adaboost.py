# coding: utf-8

import numpy as np
import pandas as pd
from sklearn.linear_model import SGDClassifier as weak_learner

def boost(X, y, sample_weights, learning_rate=.1, epochs=100, perfection=False):
    N = X.shape[0]

    classifier = weak_learner(loss='log', alpha=learning_rate, n_iter=epochs)
    
    # sklearn implementation requires y to be flattened
    y = np.ravel(y)
    
    classifier.fit(X, y, sample_weight=sample_weights)
    
    predictions = classifier.predict(X)
    print 'Predictions:', predictions
    
    indicator = (predictions != y).astype(int)
    print 'Indicator:', indicator
    
    error = float(np.dot(sample_weights, indicator))/np.sum(sample_weights)
    print error
    
    if error == 0:
        perfection = True
        alpha = float('inf')
        return classifier, sample_weights, alpha, perfection
    
    alpha = .5 * np.log((1-error)/error)
    print 'e=%.2f a=%.2f'%(error, alpha)
    
    new_sample_weights = sample_weights * np.exp(alpha*indicator)
    print 'New weights:', new_sample_weights
    
    return classifier, new_sample_weights, alpha, perfection

def fit(X, y, learning_rate=.1, epochs=100, no_predictors=5, add_bias=True):
    N = X.shape[0]
    
    sample_weights = np.array([float(1)/N,] * N)
    print 'Initial Weights:', sample_weights
    
    for iter in xrange(no_predictors):
        classifier, sample_weights, alpha, perfection = boost(X, y, sample_weights)
        if perfection:
            break
    
    return classifier
    
def main():

    #df_x = pd.read_csv('../Datasets/TrainData/X_train_0.csv', header = None)
    #x = np.array(df_x)
    #print x
    
    #df_y = pd.read_csv('../Datasets/TrainData/y_train_0.csv', header = None)
    #y = np.array(df_y)
    #print y   
    
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
    
    w = fit(x,y)
    print 'Final Classifier:', w
    
    print w.predict(x)
    return
    

if __name__ == '__main__':
    main()