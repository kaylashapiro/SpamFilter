# coding: utf-8

import importlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import metrics as met
import bagging as bag

def runTests(no_iterations, no_predictors, perc_poisoning, bagging_samples, feature_subsampling, label_switching, attack = 'Dict', classifier = 'logisticReg'):
    folder_paths = {
        'No': '../Datasets/TrainData/',
        'Dict': '../Datasets/DictAttackData/',
        'Empty': '../Datasets/EmptyAttackData/',
    }
    
    train_folder = folder_paths[attack]

    print trainBaseClassifier(no_iterations, perc_poisoning, train_folder, attack, classifier)
    
    for perc_bag in bagging_samples:
        for perc_feat in feature_subsampling:
            for perc_label in label_switching:
                print trainBaggedClassifier(no_iterations, no_predictors, perc_bag, perc_feat, perc_label, perc_poisoning, train_folder, attack, classifier)
    
            
def trainBaseClassifier(no_iterations, perc_poisoning, train_folder, attack='Dict', classifier='logisticReg'):
    try:
        learner = importlib.import_module(classifier)
    except ImportError as error:
        print error
        print "Failed to import learner module in runTests2.py"
        print "Available modules: 1) 'logisticReg' 2) 'adaline'"
        sys.exit(0)
        
    test_folder = '../Datasets/TestData/'
    
    if (perc_poisoning != 0):
        data_folder = str(perc_poisoning) + '_perc_poison/'
    else:
        data_folder = ''
    
    sum_error, sum_TPR, sum_FPR, sum_FNR, sum_TNR, sum_AUC = 0, 0, 0, 0, 0, 0
    
    for iter in xrange(no_iterations): 
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
    
        weights = learner.fit(X_train, y_train)
        predictions = learner.predict(X_test, weights)
        sum_error += met.computeError(y_test, predictions)
        sum_AUC += met.computeAUC(y_test, predictions)
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
    AUC = sum_AUC/no_iterations
    
    saveToFile(1,1,0,perc_poisoning,error,TPR,FPR,FNR,TNR,AUC,attack,classifier)
    
    return (error, TPR, FPR, FNR, TNR, AUC)

def trainBaggedClassifier(no_iterations, no_predictors, perc_instances, perc_feature_subsampling, perc_label_switching, 
                          perc_poisoning, train_folder, attack='Dict', classifier = 'logisticReg'):
    test_folder = '../Datasets/TestData/'          

    if (perc_poisoning != 0):
        data_folder = str(perc_poisoning) + '_perc_poison/'
    else:
        data_folder = ''
    
    X_train_file = 'X_train_' + str(0) + '.csv'
    y_train_file = 'y_train_' + str(0) + '.csv'
    X_test_file = 'X_test_' + str(0) + '.csv'
    y_test_file = 'y_test_' + str(0) + '.csv'
    
    df_train = pd.read_csv(train_folder + data_folder + X_train_file, header = None)
    X_train = np.array(df_train)
    
    df_train = pd.read_csv(train_folder + data_folder + y_train_file, header = None)
    y_train = np.array(df_train)
        
    df_test = pd.read_csv(test_folder + X_test_file, header = None)
    X_test = np.array(df_test)
        
    df_test = pd.read_csv(test_folder + y_test_file, header = None)
    y_test = np.array(df_test)
    
    [sum_errors, sum_TPRs, sum_FPRs, sum_FNRs, sum_TNRs, sum_AUCs] = bag.bagPredictors(X_train, y_train, X_test, y_test, no_predictors, 
                                                                                        perc_instances, perc_feature_subsampling, perc_label_switching)
    sum_errors = np.array([sum_errors])
    sum_TPRs = np.array([sum_TPRs])
    sum_FPRs = np.array([sum_FPRs])
    sum_FNRs = np.array([sum_FNRs])
    sum_TNRs = np.array([sum_TNRs])
    sum_AUCs = np.array([sum_AUCs])
    
    for iter in xrange(1,no_iterations): 
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

        [errors, TPRs, FPRs, FNRs, TNRs, AUCs] = bag.bagPredictors(X_train, y_train, X_test, y_test, no_predictors, 
                                                                    perc_instances, perc_feature_subsampling, perc_label_switching)
        sum_errors = np.concatenate((sum_errors,np.array([errors])), axis=0)                                                     
        sum_TPRs = np.concatenate((sum_TPRs,np.array([TPRs])), axis=0)
        sum_FPRs = np.concatenate((sum_FPRs,np.array([FPRs])), axis=0)    
        sum_FNRs = np.concatenate((sum_FNRs,np.array([FNRs])), axis=0)
        sum_TNRs = np.concatenate((sum_TNRs,np.array([TNRs])), axis=0)
        sum_AUCs = np.concatenate((sum_AUCs,np.array([AUCs])), axis=0)
    
    print sum_errors
    
    errors = np.mean(sum_errors, axis=0)
    TPRs = np.mean(sum_TPRs, axis=0)
    FPRs = np.mean(sum_FPRs, axis=0)
    FNRs = np.mean(sum_FNRs, axis=0)
    TNRs = np.mean(sum_TNRs, axis=0)
    AUCs = np.mean(sum_AUCs, axis=0)
    
    print errors
      
    saveToFile(perc_instances,perc_feature_subsampling,perc_label_switching,perc_poisoning,errors,TPRs,FPRs,FNRs,TNRs,AUCs,attack,classifier)
    
    return (errors, TPRs, FPRs, FNRs, TNRs)
        
def saveToFile(perc_instances, perc_feature_subsampling, perc_label_switching, perc_poisoning, 
                errors, TPRs, FPRs, FNRs, TNRs, AUCs, attack='Dict', classifier = 'logisticReg'):
    test_results = concatenateResults(errors, TPRs, FPRs, FNRs, TNRs, AUCs)            
                
    results_folder = '../Results/'
    attack_folder = '/' + attack + 'Attack/'

    if (perc_poisoning != 0):
        data_folder = str(perc_poisoning) + '_perc_poison/'
    else:
        data_folder = ''
    
    path = results_folder + classifier + attack_folder + data_folder
    
    perc_instances = str(int(perc_instances * 100))
    perc_feature_subsampling = str(int(perc_feature_subsampling * 100))
    perc_label_switching = str(int(perc_label_switching * 100))
    
    filename = perc_instances + '_' + perc_feature_subsampling + '_' + perc_label_switching + '.csv'
    
    test_header = ['Error', 'TPR', 'FPR', 'FNR', 'TNR', 'AUC']

    results = pd.DataFrame(test_results)
    results.to_csv(path + filename, index=False, header=test_header)
    
    
def concatenateResults(errors, TPRs, FPRs, FNRs, TNRs, AUCs):    
    test_results = np.column_stack((errors, TPRs, FPRs, FNRs, TNRs, AUCs))
    
    return test_results

    
def main():

    # TEST PARAMETERS
    no_iterations = 1
    no_predictors = 10
    
    attack='Dict' # Choose from 1) 'No' 2) 'Dict' 3) 'Empty'
    classifier = 'logisticRegression/' # Choose from 1) 'logisticReg', 2) 'adaline'
    
    perc_poisoning = [10, 20, 30]
    bagging_samples = [.6, .8, 1.0]
    feature_subsampling = [.5, .7, .9]
    label_switching = [0.0, 0.1, 0.2]
    # END TEST PARAMETERS
    
    for perc in perc_poisoning:
        runTests(no_iterations, no_predictors, perc, bagging_samples, feature_subsampling, label_switching, attack, classifier)
    
    
# This is the standard boilerplate that calls the main() function.
if __name__ == '__main__':
    main()