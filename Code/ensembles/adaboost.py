# coding: utf-8

import sys
import numpy as np
import pandas as pd
from sklearn.linear_model import SGDClassifier as weak_learner
from sklearn.ensemble import AdaBoostClassifier

sys.path.insert(0, '../classifiers')
sys.path.insert(0, '../helpers')

import metrics as met
from add_bias import addBias
from featureSubsampling import featureSubsampling

def boost(X, y, sample_weight, learning_rate=.1, epochs=100, terminate=False, classifier='DecisionTree'):
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
    - terminate: boolean termination condition
    
    Output:
    - classifier: trained classifier object
    - new_sample_weight: 1 * N Numpy vector of sample weightings
    - alpha: weight of the classifier prediction
    - terminate: boolean termination condition
    '''
    N = X.shape[0]

    classifier = weak_learner(loss='log', penalty='none')
    
    # sklearn implementation requires y to be flattened
    y = np.ravel(y)
    sample_weights = sample_weight/np.sum(sample_weight)    
        
    # 1. Train weak learner with sample_weights distrubtion    
    classifier.fit(X, y, sample_weight=sample_weight)
    
    predictions = classifier.predict(X)
    print 'Predictions:', predictions    
    
    indicator = (predictions != y).astype(int)
    print 'Indicator:', indicator
    
    # 2. Measure the error based on the probability distribution
    error = np.dot(sample_weight, indicator)
    
    # 3. If error == 0, we fit the data perfectly
    if error == 0:
        terminate = True
        alpha = float('inf')
        return classifier, sample_weight, alpha, terminate
    
    # 4. Determine the weight of classifier
    alpha = .5 * np.log((1-error)/error)
    print 'e=%.5f a=%.5f'%(error, alpha)
    
    # 5. Update the distribution
    new_sample_weight = sample_weight * np.exp(-alpha*indicator)
    print 'New weights:', new_sample_weight
    
    return classifier, new_sample_weight, alpha, terminate

def fit(X, y, X_test, y_test, 
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
    
    Output:
    - classifiers: list of classifier objects at each boosting 
                   iteration
    - alphas: weights of each classifier's prediction
    '''
    N = X.shape[0]
    classifiers = []
    alphas = []
    errors = []
    TPRs = []
    FPRs = []
    FNRs = []
    TNRs = []
    AUCs = []
    
    # 0. Initialize uniform weights
    sample_weight = np.array([float(1)/N,] * N)
    print 'Initial Weights:', sample_weight
    
    for iter in xrange(no_predictors):
        classifier, sample_weight, alpha, terminate = boost(X, y, sample_weight)
        classifiers.append(classifier)
        alphas.append(alpha)
        
        predictions = predict(X_test, classifiers, alphas)
        
        errors.append(met.computeError(y_test, predictions))
        
        AUCs.append(met.computeAUC(y_test, predictions))
                
        [TPR, FPR, FNR, TNR] = met.computeRates(y_test, predictions, ham_label, spam_label)
        
        TPRs.append(TPR)
        FPRs.append(FPR)
        FNRs.append(FNR)
        TNRs.append(TNR)
                
        if terminate:
            return (classifiers, alphas, errors, TPRs, FPRs, FNRs, TNRs, AUCs)
    
    return (classifiers, alphas, errors, TPRs, FPRs, FNRs, TNRs, AUCs)
    
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
    
    
    
def main():

    
    df_x = pd.read_csv('../../Datasets/TrainData/enron/X_train_0.csv', header = None)
    x = np.array(df_x)
    print x
    
    x_train, x_test = featureSubsampling(x, x, .7)
    x_train = addBias(x_train)
    x_test = addBias(x_test)
    
    df_y = pd.read_csv('../../Datasets/TrainData/enron/y_train_0.csv', header = None)
    y = np.ravel(np.array(df_y))
    print y   
    
    # BASE CLASSIFIER
    classifier = weak_learner(loss='log', penalty='none')
    classifier.fit(x, y)
    predictions = classifier.predict(x)
    print 'BASE CLASSIFIER ERROR:', met.computeError(y, predictions)
    
    
    classifiers, alphas, errors, TPRs, FPRs, FNRs, TNRs, AUCs = fit(x,y,x,y,no_predictors=10)
    print 'Final Classifier:', classifiers[-1] 
    
    predictions = predict(x, classifiers, alphas)
    print 'Final Predictions:', predictions
    
    print 'Errors:', errors
    print 'TPRs:', TPRs
    print 'FPRs:', FPRs
    print 'FNRs:', FNRs
    print 'TNRs:', TNRs
    
    error = met.computeError(y, predictions)
    print 'Final Error:', error
    
    classifier = AdaBoostClassifier(n_estimators=50, learning_rate=.1)
    classifier.fit(x,np.ravel(y))
    pred = classifier.predict(x)
    print pred
    err = met.computeError(y, pred)
    print 'SKLEARN ERROR:', err
    
    return

if __name__ == '__main__':
    main()