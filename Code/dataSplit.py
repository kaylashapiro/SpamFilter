# coding: utf-8

import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split
import random

def dataSplit(X, y, no_iterations=10, dataset='enron'):
    '''
    Splits processed features and labels into different training and test sets
    for a certain number of iterations and saves them to .csv files
    
    Inputs:
    - X: N * D Numpy matrix of binary feature values (0 and 1)
         with N: the number of training examples
         and  D: the number of features for each example
    - y: N * 1 Numpy vector of binary values (0 and 1); training set
    -no_iterations: Integer number of training sets to iterate through and attacks
                    to generate at each perc_poisoning
                    
    Output:
    NONE
    '''

    rand_no = random.randint(0,1000) # Generate random state
    size = .5
    path_train = '../Datasets/TrainData/' + dataset + '/'
    path_test = '../Datasets/TestData/' + dataset + '/'
    
    for iter in xrange(no_iterations):
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=size,random_state=rand_no)
        
        X_train_name = 'X_train_' + str(iter) + '.csv'
        X_test_name = 'X_test_' + str(iter) + '.csv'
        y_train_name = 'y_train_' + str(iter) + '.csv'
        y_test_name = 'y_test_' + str(iter) + '.csv'
        
        my_X_train = pd.DataFrame(X_train, dtype = np.uint8)
        my_X_train.to_csv(path_train + X_train_name, index=False, header=False)
    
        my_X_test = pd.DataFrame(X_test, dtype = np.uint8)
        my_X_test.to_csv(path_test + X_test_name, index=False, header=False)
    
        my_y_train = pd.DataFrame(y_train, dtype = np.uint8)
        my_y_train.to_csv(path_train + y_train_name, index=False, header=False)
    
        my_y_test = pd.DataFrame(y_test, dtype = np.uint8)
        my_y_test.to_csv(path_test + y_test_name, index=False, header=False)
    
    return 
    
# Main function to run algorithm on various fractions of attacker knowledge and control.
def main():
    dataset = 'lingspam'
    features_path = '../Datasets/EmailDataProcessed/' + dataset + '/Features.csv'
    labels_path = '../Datasets/EmailDataProcessed/' + dataset + '/Labels.csv'

    df_X = pd.read_csv(features_path, header = None)
    X = np.array(df_X)
    print X
    
    df_y = pd.read_csv(labels_path, header = None)
    y = np.array(df_y)
    print y
    
    dataSplit(X,y,dataset=dataset)
    
    
# This is the standard boilerplate that calls the main() function.
if __name__ == '__main__':
    main()