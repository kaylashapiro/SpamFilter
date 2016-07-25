# coding: utf-8

import importlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import metrics as met
import bagging as bag

def runPoisonTests():
    classifier = 'logisticReg'
    attack = 'Dict'
    perc_poisoning = [10, 20, 30]
    bagging_samples = [.6, .8, 1.0]
    feature_subsampling = [.5, .7, .9]
    label_switching = [0.0, 0.1, 0.2]

def runCleanTests(no_iterations, no_predictors):
    classifier = 'logisticReg'
    attack = 'No'
    perc_poisoning = 0
    bagging_samples = [.6, .8, 1.0]
    feature_subsampling = [.5, .7, .9]
    label_switching = [0.0, 0.1, 0.2]
    
    trainBaseClassifier()
    
    for perc_bag in bagging_samples:
        for perc_feat in feature_subsampling:
            for perc_label in label_switching:
                x = 1
            
def trainBaseClassifier(no_iterations, perc_poisoning, attack='Dict', classifier='logisticReg'):
    try:
        learner = importlib.import_module(classifier)
    except ImportError as error:
        print error
        print "Failed to import learner module in runTests2.py"
        print "Available modules: 1) 'logisticReg' 2) 'adaline'"
        sys.exit(0)
    
    folder_paths = {
        'No': '../Datasets/TrainData/',
        'Dict': '../Datasets/DictAttackData/',
        'Empty': '../Datasets/EmptyAttackData/',
    }
        
    train_folder = folder_paths[attack]
    test_folder = '../Datasets/TestData/'
    
    if (perc_poisoning != 0):
        data_folder = str(perc) + '_perc_poison/'
    else:
        data_folder = ''
    
    sum_error, sum_TPR, sum_FPR, sum_FNR, sum_TNR = 0, 0, 0, 0, 0
    
    for iter in xrange(no_iterations): 
        X_train_file = 'X_train_' + str(iter) + '.csv'
        y_train_file = 'y_train_' + str(iter) + '.csv'
        X_test_file = 'X_test_' + str(iter) + '.csv'
        y_test_file = 'y_test_' + str(iter) + '.csv'
    
        df_train = pd.read_csv(train_folder + data_folder + X_train_file, header = None)
        X_train = np.array(df_train)
    
        df_train = pd.read_csv(train_folder + data_folder + y_train_file, header = None)
        y_train = np.array(df_train)
        
        df_test = pd.read_csv(test_folder + data_folder + X_test_file, header = None)
        X_test = np.array(df_test)
        
        df_test = pd.read_csv(test_folder + data_folder + y_test_file, header = None)
        y_test = np.array(df_test)
    
        weights = learner.fit(X_train, y_train)
        predictions = learner.predict(X_test, weights)
        sum_error += met.computeError(predictions, y_test)
        [TP, FP, FN, TN] = met.computeMetrics(y_test, predictions)
        [TPR, FPR, FNR, TNR] = met.computeRates(TP, FP, FN, TN)
        sum_TPR += TPR
        sum_FPR += FPR
        sum_FNR += FNR
        sum_TNR = TNR
        
    error = sum_error/no_iterations # ADD DIVIDE BY ZERO EXCEPTION OR DO A CHECK SOMEWHERE ELSE FOR NO_ITERATIONS
    TPR = sum_TPR/no_iterations
    FPR = sum_FPR/no_iterations
    FNR = sum_FNR/no_iterations
    TNR = sum_TNR/no_iterations
    
    saveToFile(1,1,0,perc_poisoning, error, TPR, FPR, FNR, TNR, classifier, attack)
    
    return (error, TPR, FPR, FNR, TNR)
    
def trainBaggedClassifier(X_train, y_train, X_test, y_test, no_predictors, 
                            perc_instances, perc_feature_subsampling, perc_label_switching):
    [error, TPRs, FPRs, FNRs, TNRs] = bag.bagPredictors(X_train, y_train, X_test, y_test, no_predictors, 
                                                        perc_instances, perc_feature_subsampling, perc_label_switching)
    
    return (error, TPR, FPR, FNR, TNR)
        
def saveToFile(perc_instances, perc_feature_subsampling, perc_label_switching, perc_poisoning, 
                errors, TPRs, FPRs, FNRs, TNRs, classifier = 'logisticReg', attack='Dict'):
    test_results = concatenateResults(errors, TPRs, FPRs, FNRs, TNRs)            
                
    results_folder = '../Results/'
    attack_folder = '/' + attack + 'Attack/'

    if (perc_poisoning != 0):
        data_folder = str(perc) + '_perc_poison/'
    else:
        data_folder = ''
    
    path = results_folder + classifier + attack_folder + data_folder
    
    perc_instances = str(int(perc_instances * 100))
    perc_feature_subsampling = str(int(perc_feature_subsampling * 100))
    perc_label_switching = str(int(perc_label_switching * 100))
    
    filename = perc_instances + '_' + perc_feature_subsampling + '_' + perc_label_switching + '.csv'
    
    test_header = ['Error', 'TPR', 'FPR', 'FNR', 'TNR']

    results = pd.DataFrame(test_results)
    results.to_csv(path + filename, index=False, header=test_header)
    
    
def concatenateResults(errors, TPRs, FPRs, FNRs, TNRs):    
    test_results = np.column_stack((errors, TPRs, FPRs, FNRs, TNRs))
    
    return test_results

    
def main():
    # THESE ARE THE SETTINGS WE WANT
    #no_iterations = 10
    #no_predictors = 80
    #frac_knowl = 1
    #bagging_samples = [.6, .8, 1.0]
    #feature_subsampling = [.5, .7, .9]
    #perc_poisoning = [10, 20, 30]
    #label_switching = [0.0, 0.1, 0.2]
    # END SETTINGS

    # TEST PARAMETERS
    classifier = 'logisticRegression/' # Choose from 1) 'logisticReg', 2) 'adaline'
    attack='Dict'
    no_iterations = 1
    no_predictors = 10
    frac_knowl = 1
    bagging_samples = 1
    perc_poisoning = 30
    label_switching = .2
    # END TEST PARAMETERS
    
    errors = [1,2,3]
    TPRs = [4,5,6]
    FPRs = [7,8,9]
    FNRs = [3,2,1]
    TNRs = [6,5,4]
    
    #saveToFile(.1, .1, .1, 10, errors, TPRs, FPRs, FNRs, TNRs)
    
    print trainBaseClassifier(2, 0, attack='No')
    
    
# This is the standard boilerplate that calls the main() function.
if __name__ == '__main__':
    main()