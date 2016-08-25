# coding: utf-8

import sys
import importlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, 'classifiers')
sys.path.insert(0, 'helpers')
sys.path.insert(0, 'ensembles')

import metrics as met
import bagging as bag
from add_bias import addBias

def runTests(no_iterations, no_predictors, perc_poisoning, bagging_samples, feature_subsampling, label_switching, 
             attack='Dict', 
             classifier='logistic_regression',
             dataset=None):
    '''
    Inputs:
    - no_iterations: integer number of experiments to run on a given test set-up; results of the experiments are
                     averaged
    - no_predictors: integer number of classifiers to bag
    - perc_poisoning: integer between 0 and 100; percentage of poisoning of the training set
    - bagging_samples: real number fraction of the training set to put into a bootstrap replicate set; .5 = 50% of 
                       size of the training set, 1 = 100% of size of the training set, 2 = 200% of size of the training
                       set
    - feature_subsampling: real number between 0 and 1; fraction of features to subsample
    - label_switching: real number between 0 and 1; fraction of labels to switch
    - attack: string; Choose from: 1) 'Dict', 2) 'Empty', 3) 'Ham', 4) 'Optimal'
    - classifier: string; Choose from: 1) 'logisticReg', 2) 'adaline', 3) 'naivebayes'
    - dataset: string; choose from 1) 'enron' 2) 'lingspam'
    
    Ouput:
    NONE
    '''
    
    if no_iterations <= 0:
        print 'Need at least 1 experiment iteration.'
        sys.exit(0)
    if no_predictors <= 0:
        print 'Need at least 1 predictor to run experiment.'
        sys.exit(0)
    
    folder_paths = {
        'No': '../Datasets/TrainData/',
        'Dict': '../Datasets/DictAttackData/',
        'Empty': '../Datasets/EmptyAttackData/',
        'Ham': '../Datasets/HamAttackData/',
        'Ham2': '../Datasets/Ham2AttackData/'
    }
    
    ## path to access the training data
    train_folder = folder_paths[attack] + dataset + '/'
    
    ## path to access the test data
    test_folder = '../Datasets/TestData/' + dataset + '/'          

    ## access/store data based on percentage of poisoning
    if (perc_poisoning != 0):
        data_folder = str(perc_poisoning) + '_perc_poison/'
    else:
        data_folder = ''

    ## train a single classifier at given poisoning level for a baseline
    print trainBaseClassifier(no_iterations, perc_poisoning, train_folder, test_folder, data_folder, attack, classifier, dataset=dataset)
    
    ## train a generic bagging classifier without feature_subsampling and label_switching
    print trainBaggedClassifier(no_iterations, no_predictors, 1, 1, 0, perc_poisoning, train_folder, test_folder, data_folder, attack, classifier, dataset=dataset)
    
    ## train bagged classifiers with feature_subsampling and label_switching
    for perc_bag in bagging_samples:
        for perc_feat in feature_subsampling:
            for perc_label in label_switching:
                print trainBaggedClassifier(no_iterations, no_predictors, perc_bag, perc_feat, perc_label, perc_poisoning, train_folder, test_folder, data_folder, attack, classifier, dataset=dataset)
    return
    
            
def trainBaseClassifier(no_iterations, perc_poisoning, train_folder, test_folder, data_folder, 
                        attack='Dict', 
                        classifier='logistic_regression',
                        dataset=None,
                        ham_label=0,
                        spam_label=1):
    '''
    Inputs:
    - no_iterations: integer number of experiments to run on a given test set-up; results of the experiments are
                     averaged
    - perc_poisoning: integer between 0 and 100; percentage of poisoning of the training set
    - train_folder: string; path to correct attack folder training sets
    - test_folder: string; path to correct test set folder
    - data_folder: string; folder for a given percentage of poisoning; empty for 'NoAttack'
    - attack: string; Choose from: 1) 'Dict', 2) 'Empty', 3) 'Ham', 4) 'Optimal'
    - classifier: string; Choose from: 1) 'logisticReg', 2) 'adaline', 3) 'naivebayes'
    - dataset: string; choose from 1) 'enron' 2) 'lingspam'
    
    Ouput:
    - error: error value
    - TPR: true positive rate
    - FPR: false positive rate
    - FNR: false negative rate
    - TNR: true negative rate
    - AUC: AUC value (see sklearn.metrics.roc_auc_score documentation)
    '''
    ## import the classifier that we are going to train
    try:
        learner = importlib.import_module(classifier)
    except ImportError as error:
        print error
        print "Failed to import learner module in run_tests.py"
        print "Available modules: 1) 'logisticReg' 2) 'adaline'"
        sys.exit(0)
    
    ## initialize metrics
    sum_error, sum_TPR, sum_FPR, sum_FNR, sum_TNR, sum_AUC = 0, 0, 0, 0, 0, 0
    
    for iter in xrange(no_iterations): 
        print 'STARTING ITER:', iter
        X_train_file = 'X_train_' + str(iter) + '.csv'
        y_train_file = 'y_train_' + str(iter) + '.csv'
        X_test_file = 'X_test_' + str(iter) + '.csv'
        y_test_file = 'y_test_' + str(iter) + '.csv'
    
        df_train = pd.read_csv(train_folder + data_folder + X_train_file, header = None)
        X_train = np.array(df_train)
    
        df_train = pd.read_csv(train_folder + data_folder + y_train_file, header = None)
        y_train = np.array(df_train)
        
        df_test = pd.read_csv(test_folder + X_test_file, header = None)
        X_test = np.array(df_test)
        
        df_test = pd.read_csv(test_folder + y_test_file, header = None)
        y_test = np.array(df_test)
        
        if classifier is not 'naivebayes':
            X_train = addBias(X_train)
            X_test = addBias(X_test)
    
        ## train the classifier and make predictions on the test set
        weights = learner.fit(X_train, y_train)
        predictions = learner.predict(X_test, weights)
        
        ## record the metrics for this iteration
        sum_error += met.computeError(y_test, predictions)
        sum_AUC += met.computeAUC(y_test, predictions)
        [TPR, FPR, FNR, TNR] = met.computeRates(y_test, predictions, ham_label, spam_label)
        
        sum_TPR += TPR
        sum_FPR += FPR
        sum_FNR += FNR
        sum_TNR += TNR
        
    ## take the average of all the metrics
    error = sum_error/no_iterations
    TPR = sum_TPR/no_iterations
    FPR = sum_FPR/no_iterations
    FNR = sum_FNR/no_iterations
    TNR = sum_TNR/no_iterations
    AUC = sum_AUC/no_iterations
    
    # arguments 0,0,0 signal base classifier
    saveToFile(0,0,0,perc_poisoning,error,TPR,FPR,FNR,TNR,AUC,attack,classifier,dataset=dataset)
    
    return (error, TPR, FPR, FNR, TNR, AUC)

def trainBaggedClassifier(no_iterations, no_predictors, perc_instances, perc_feature_subsampling, perc_label_switching, perc_poisoning, train_folder, test_folder, data_folder,
                          attack='Dict', 
                          classifier = 'logisticReg',
                          dataset=None):
    '''
    Inputs:
    - no_iterations: integer number of experiments to run on a given test set-up; results of the experiments are
                     averaged
    - no_predictors: integer number of classifiers to bag
    - perc_instances: real number fraction of the training set to put into a bootstrap replicate set; .5 = 50% of 
                      size of the training set, 1 = 100% of size of the training set, 2 = 200% of size of the training
                      set
    - perc_feature_subsampling: real number between 0 and 1; fraction of features to subsample
    - perc_label_switching: real number between 0 and 1; fraction of labels to switch
    - perc_poisoning: integer between 0 and 100; percentage of poisoning of the training set
    - train_folder: string; path to correct attack folder training sets
    - test_folder: string; path to correct test set folder
    - data_folder: string; folder for a given percentage of poisoning; empty for 'NoAttack'
    - attack: string; Choose from: 1) 'Dict', 2) 'Empty', 3) 'Ham', 4) 'Optimal'
    - classifier: string; Choose from: 1) 'logisticReg', 2) 'adaline', 3) 'naivebayes'
    - dataset: string; choose from 1) 'enron' 2) 'lingspam'
    
    Ouputs:
    - errors: 1 * N Numpy array of error values
              with N: number of classifiers bagged
    - TPRs: 1 * N Numpy array of true positive rates
    - FPRs: 1 * N Numpy array of false positive rates
    - FNRs: 1 * N Numpy array of false negative rates
    - TNRs: 1 * N Numpy array of true negative rates
    - AUCs: 1 * N Numpy array of AUC values (see sklearn.metrics.roc_auc_score documentation)
    '''
    
    X_train_file = 'X_train_' + str(0) + '.csv'
    y_train_file = 'y_train_' + str(0) + '.csv'
    X_test_file = 'X_test_' + str(0) + '.csv'
    y_test_file = 'y_test_' + str(0) + '.csv'
    
    df_X_train = pd.read_csv(train_folder + data_folder + X_train_file, header = None)
    X_train = np.array(df_X_train)
    
    df_y_train = pd.read_csv(train_folder + data_folder + y_train_file, header = None)
    y_train = np.array(df_y_train)
        
    df_X_test = pd.read_csv(test_folder + X_test_file, header = None)
    X_test = np.array(df_X_test)
        
    df_y_test = pd.read_csv(test_folder + y_test_file, header = None)
    y_test = np.array(df_y_test)
    
    if classifier is not 'naivebayes':
        X_train = addBias(X_train)
        X_test = addBias(X_test)
    
    ## bag the classifier for a given training and test set
    [sum_errors, sum_TPRs, sum_FPRs, sum_FNRs, sum_TNRs, sum_AUCs] = bag.bagPredictors(X_train, y_train, X_test, y_test, no_predictors, 
                                                                                        perc_instances, perc_feature_subsampling, perc_label_switching, classifier)
    ## record the metrics
    sum_errors = np.array([sum_errors])
    sum_TPRs = np.array([sum_TPRs])
    sum_FPRs = np.array([sum_FPRs])
    sum_FNRs = np.array([sum_FNRs])
    sum_TNRs = np.array([sum_TNRs])
    sum_AUCs = np.array([sum_AUCs])
    
    for iter in xrange(1,no_iterations): 
        print 'STARTING ITER:', iter
        X_train_file = 'X_train_' + str(iter) + '.csv'
        y_train_file = 'y_train_' + str(iter) + '.csv'
        X_test_file = 'X_test_' + str(iter) + '.csv'
        y_test_file = 'y_test_' + str(iter) + '.csv'
    
        df_train = pd.read_csv(train_folder + data_folder + X_train_file, header = None)
        X_train = np.array(df_train)
    
        df_train = pd.read_csv(train_folder + data_folder + y_train_file, header = None)
        y_train = np.array(df_train)
        
        df_test = pd.read_csv(test_folder + X_test_file, header = None)
        X_test = np.array(df_test)
        
        df_test = pd.read_csv(test_folder + y_test_file, header = None)
        y_test = np.array(df_test)
        
        if classifier is not 'naivebayes':
            X_train = addBias(X_train)
            X_test = addBias(X_test)
        
        ## bag the classifier for a given training and test set
        [errors, TPRs, FPRs, FNRs, TNRs, AUCs] = bag.bagPredictors(X_train, y_train, X_test, y_test, no_predictors, 
                                                                    perc_instances, perc_feature_subsampling, perc_label_switching, classifier)
        ## record the metrics
        sum_errors = np.concatenate((sum_errors,np.array([errors])), axis=0)                                                     
        sum_TPRs = np.concatenate((sum_TPRs,np.array([TPRs])), axis=0)
        sum_FPRs = np.concatenate((sum_FPRs,np.array([FPRs])), axis=0)    
        sum_FNRs = np.concatenate((sum_FNRs,np.array([FNRs])), axis=0)
        sum_TNRs = np.concatenate((sum_TNRs,np.array([TNRs])), axis=0)
        sum_AUCs = np.concatenate((sum_AUCs,np.array([AUCs])), axis=0)
    
    ## take the average of all the metrics
    errors = np.mean(sum_errors, axis=0)
    TPRs = np.mean(sum_TPRs, axis=0)
    FPRs = np.mean(sum_FPRs, axis=0)
    FNRs = np.mean(sum_FNRs, axis=0)
    TNRs = np.mean(sum_TNRs, axis=0)
    AUCs = np.mean(sum_AUCs, axis=0)
    
    ## save experiment
    saveToFile(perc_instances,perc_feature_subsampling,perc_label_switching,perc_poisoning,errors,TPRs,FPRs,FNRs,TNRs,AUCs,attack,classifier,dataset=dataset)
    
    return (errors, TPRs, FPRs, FNRs, TNRs, AUCs)
        
def saveToFile(perc_instances, perc_feature_subsampling, perc_label_switching, perc_poisoning, errors, TPRs, FPRs, FNRs, TNRs, AUCs, 
                attack='Dict', 
                classifier = 'logistic_regression',
                dataset=None):
    '''
    Inputs:
    - perc_instances: real number fraction of the training set to put into a bootstrap replicate set; .5 = 50% of 
                      size of the training set, 1 = 100% of size of the training set, 2 = 200% of size of the training
                      set
    - perc_feature_subsampling: real number between 0 and 1; fraction of features to subsample
    - perc_label_switching: real number between 0 and 1; fraction of labels to switch
    - perc_poisoning: integer between 0 and 100; percentage of poisoning of the training set
    - errors: 1 * N Numpy array of error values
              with N: number of classifiers bagged
    - TPRs: 1 * N Numpy array of true positive rates
    - FPRs: 1 * N Numpy array of false positive rates
    - FNRs: 1 * N Numpy array of false negative rates
    - TNRs: 1 * N Numpy array of true negative rates
    - AUCs: 1 * N Numpy array of AUC values (see sklearn.metrics.roc_auc_score documentation)
    - attack: string; Choose from: 1) 'Dict', 2) 'Empty', 3) 'Ham', 4) 'Optimal'
    - classifier: string; Choose from: 1) 'logisticReg', 2) 'adaline', 3) 'naivebayes'
    - dataset: string; choose from 1) 'enron' 2) 'lingspam'
    
    Ouput:
    NONE
    '''              
    test_results = concatenateResults(errors, TPRs, FPRs, FNRs, TNRs, AUCs)            
                
    results_folder = '../Results/'
    attack_folder = '/' + attack + 'Attack/'
    dataset_folder = '/' + dataset 

    if (perc_poisoning != 0):
        data_folder = str(perc_poisoning) + '_perc_poison/'
    else:
        data_folder = ''
    
    path = results_folder + classifier + dataset_folder + attack_folder + data_folder
    
    perc_instances = str(int(perc_instances * 100))
    perc_feature_subsampling = str(int(perc_feature_subsampling * 100))
    perc_label_switching = str(int(perc_label_switching * 100))
    
    filename = perc_instances + '_' + perc_feature_subsampling + '_' + perc_label_switching + '.csv'
    
    test_header = ['Error', 'TPR', 'FPR', 'FNR', 'TNR', 'AUC']

    results = pd.DataFrame(test_results)
    results.to_csv(path + filename, index=False, header=test_header)
    
    return
    
    
def concatenateResults(errors, TPRs, FPRs, FNRs, TNRs, AUCs):    
    '''
    Inputs:
    - errors: 1 * N Numpy array of error values
              with N: number of classifiers bagged
    - TPRs: 1 * N Numpy array of true positive rates
    - FPRs: 1 * N Numpy array of false positive rates
    - FNRs: 1 * N Numpy array of false negative rates
    - TNRs: 1 * N Numpy array of true negative rates
    - AUCs: 1 * N Numpy array of AUC values (see sklearn.metrics.roc_auc_score documentation)
    
    Output:
    - test_results: N * 6 Numpy matrix with columns =[errors.T, TPRs.T, FPRs.T, FNRs.T, TNRs.T, AUCs.T]
    '''
    
    test_results = np.column_stack((errors, TPRs, FPRs, FNRs, TNRs, AUCs))
    
    return test_results

    
def main():

    # TEST PARAMETERS
    no_iterations = 5
    no_predictors = 60
    
    ## SELECT DATASET
    dataset='enron'
    
    ## SELECT ATTACK ('No', 'Dict', 'Empty', 'Ham', 'Ham2')
    attack='Dict'
    
    ## SELECT CLASSIFIER ('logistic_regression', 'adaline', 'naivebayes')
    classifier = 'adaline_with_adadelta'
    
    # SELECT PERCENT OF POISONING
    #perc_poisoning = [0] # No Attack
    #perc_poisoning = [10, 20, 30] # Attack
    perc_poisoning = [20]
    
    # BAGGING PARAMETERS
    bagging_samples = [.6, .8, 1.0]
    feature_subsampling = [.5, .7, .9]
    label_switching = [0.0, 0.1, 0.2]
    
    for perc in perc_poisoning:
        runTests(no_iterations, no_predictors, perc, bagging_samples, feature_subsampling, label_switching, attack, classifier,dataset=dataset)
    
# This is the standard boilerplate that calls the main() function.
if __name__ == '__main__':
    main()
