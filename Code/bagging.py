# This is an implementation of a bagging algorithm
# Attacker knowledge references the fraction of features an attacker knows

import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import logisticRegVec as lr
import sampleWithReplacement as swr
from sklearn.metrics import roc_curve, auc
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression


def bagPredictors(X_train, y_train, X_test, y_test, no_predictors):

    no_instances = X_train.shape[0]
    
    base_classifier = LogisticRegression(max_iter=1000)
    base_classifier.fit(X_train, y_train)
    predictions = base_classifier.predict(X_test)
    #print 'Base Error:', computeError(predictions, y_test)
    
    new_instances = 10 # Should be same number of instances as X
    errors = []

    replicate, labels = swr.generateReplicate(X_train, y_train, no_instances)
    #print replicate
    #print 'Labels', labels
    
    #thetas = [lr.regLogisticRegression(replicate, labels)]
    
    classifier = LogisticRegression(max_iter=1000)    
    
    classifier.fit(replicate, labels)    
    
    predictions = classifier.predict(X_test)
    
    votes = predictions
    
    #print 'Single Classifier Predictions', predictions
    
    errors.append(computeError(predictions, y_test))
    
    for ith_predictor in range(1, no_predictors):
        replicate, labels = swr.generateReplicate(X_train, y_train, no_instances)
        #print replicate
        #print 'Labels', labels
        
        classifier.fit(replicate, labels)
        
        predictions = classifier.predict(X_test)
        #print 'Single Classifier Predictions', predictions
        
        #print computeError(predictions, y_test)
        
        votes += predictions
        #print 'Votes', votes
        
        predictions = computeClass(votes, ith_predictor + 1)
        #print 'Actual Predictions', predictions
        
        errors.append(computeError(predictions, y_test))
        #print errors
               
    return errors

def computeClass(votes, ith_predictor):

    votes = votes/float(ith_predictor)
    #print 'computeClass', votes
    
    votes[votes<=.5] = 0
    votes[votes>.5] = 1
    
    return votes
    
def computeError(predictions, y_test):
    return np.mean(predictions != y_test)

# Main function to run algorithm on various fractions of attacker knowledge and control.
def main():
    df_X = pd.read_csv('Features.csv', header = None)
    X = np.array(df_X)
    print X
    
    df_y = pd.read_csv('Labels.csv', header = None)
    y = np.array(df_y).T[0]
    print y
    
    no_predictors = 3
    
    print bagPredictors(X, y, X, y, no_predictors)
    
    
# This is the standard boilerplate that calls the main() function.
if __name__ == '__main__':
    main()