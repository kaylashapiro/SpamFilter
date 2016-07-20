import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import dictionaryAttack as attack
import bagging as bag
import logisticReg as lr
from sklearn.metrics import roc_curve, auc
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression

def computeError(predictions, y_test):
    return np.mean(predictions != y_test)
    
def computeAverageError(errors):
    return errors.mean(axis=0)

def trainBaseClassifier(X_train, y_train, X_test, y_test, no_iterations):
    error, tot_TPR, tot_FPR, tot_FNR, tot_TNR = [0,0,0,0,0]
    
    for i in xrange(no_iterations):
        weights = lr.fit(X_train, y_train)
        predictions = lr.predict(X_test, weights)
        
        error += computeError(predictions, y_test)
        [TP, FP, FN, TN] = lr.computeMetrics(y_test, predictions)
        [TPR, FPR, FNR, TNR] = lr.computeRates(TP, FP, FN, TN)
        tot_TPR += TPR
        tot_FPR += FPR
        tot_FNR += FNR
        tot_TNR += TNR
    error = error/no_iterations
    TPR = tot_TPR/no_iterations
    FPR = tot_FPR/no_iterations
    FNR = tot_FNR/no_iterations
    TNR = tot_TNR/no_iterations
    
    return (error, TPR, FPR, FNR, TNR)


# Main function to run the dictionary attack
def main():
    # TASK 0: SET TEST PARAMETERS
    frac_knowl = 1 # optimal attack
    frac_mal_instances = .3
    no_predictors = 80
    no_features = 1000
    no_iterations = 10

    
    # TASK 1: Read binary feature set from 'Features.csv' and 'Labels.csv'
    df_X = pd.read_csv('Features.csv', header = None)
    X = np.array(df_X)
    #print X
    
    df_y = pd.read_csv('Labels.csv', header = None)
    y = np.array(df_y)
    #print y
    
    
    # TASK 2: Split data set into training and test sets using sklearn.cross_validation.train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    #print X_train
    #print X_test
    
    
    # TASK 3 (OPTIONAL): Poison the training set with one of the following attacks: 
    # 1) Empty Attack, 2) Dictionary Attack, 3) Ham Attack, 4) Focused Attack
    [X_train_poisoned, y_train_poisoned] = attack.poisonData(X, y, frac_knowl, frac_mal_instances)
    #print X_train_poisoned
    #print y_train_poisoned
    
    
    # TASK 4: Train a base linear classifier on clean data
    [base_error, base_TPR, base_FPR, base_FNR, base_TNR] = trainBaseClassifier(X_train, y_train, X_test, y_test, no_iterations)
    
    print 'base linear classifier on clean data:', base_error
    print 'TPR:', base_TPR
    print 'FPR:', base_FPR
    print 'FNR:', base_FNR
    print 'TNR:', base_TNR
    
    
    # TASK 5: Train a base linear classifier on poisoned data
    [base_error_poisoned, base_TPR_p, base_FPR_p, base_FNR_p, base_TNR_p] = trainBaseClassifier(X_train_poisoned, y_train_poisoned, X_test, y_test, no_iterations)
    print 'base linear classifier on poisoned data:', base_error_poisoned  
    print 'TPR:', base_TPR_p
    print 'FPR:', base_FPR_p
    print 'FNR:', base_FNR_p
    print 'TNR:', base_TNR_p
    
    
    # TASK 6: Train no_iterations bagged linear classifiers on clean data
    bag_clean_errors = np.array([bag.bagPredictors(X_train, y_train, X_test, y_test, no_predictors)])
    #print bag_clean_errors
    for i in xrange(1, no_iterations):
        bag_clean_errors = np.concatenate((bag_clean_errors, np.array([bag.bagPredictors(X_train, y_train, X_test, y_test, no_predictors)])), axis=0)
        #print bag_clean_errors
    #print bag_clean_errors
    bag_clean_errors = np.array(computeAverageError(bag_clean_errors))
    #print bag_clean_errors
    #print bag_clean_errors.shape
    
    
    #TASK 7: Train no_iterations bagged linear classifiers on poisoned data
    bag_poisoned_errors = np.array([bag.bagPredictors(X_train_poisoned, y_train_poisoned, X_test, y_test, no_predictors)])
    for i in xrange(1, no_iterations):
        bag_poisoned_errors = np.concatenate((bag_poisoned_errors,np.array([bag.bagPredictors(X_train_poisoned, y_train_poisoned, X_test, y_test, no_predictors)])), axis=0)
        #print bag_poisoned_errors
    #print bag_poisoned_errors
    bag_poisoned_errors = np.array(computeAverageError(bag_poisoned_errors))
    #print bag_poisoned_errors
    #print bag_poisoned_errors.shape
    
    # TASK 8: Plot the results
    # y-axis: Average classification error over 20 repititions
    # x-axis: Number of baggers (base classifiers represented by horizontal lines)
    
    x_axis = np.linspace(1,no_predictors, num=no_predictors, endpoint=True)
    base_clean = base_error * np.ones(len(x_axis))
    base_poisoned = base_error_poisoned * np.ones(len(x_axis))
    
    plt.figure(figsize=(10,6), dpi=120)
        
    plt.plot(x_axis, base_clean, 'b--', label='Logistic Regression')
    plt.plot(x_axis, base_poisoned, 'r--', label = 'Logistic Regression (poisoned)')
    plt.plot(x_axis, bag_clean_errors, color='blue', label='Bagged')
    plt.plot(x_axis, bag_poisoned_errors, color='red', label = 'Bagged (poisoned)')
    
    plt.title('Dictionary Attack')
    plt.xlabel('Number of Baggers')
    plt.ylabel('Error Rate')
    
    plt.xlim(0, no_predictors + 1)
    plt.ylim(.05, .1)
    
    plt.text(.035, .035, 'Using sklearn implementation of logistic regression at 1000 features without\nmutual information implementation.\nExperiment iterations: ' + str(no_iterations))
    
    plt.legend(loc='lower left', frameon=True)
    
    #plt.show()
    
    figure_name = 'LogisticRegression_' + str(no_iterations) + '_iterations_' + str(no_features) + '_features' 
    
    plt.savefig(figure_name, bbox_inches='tight')
    

# This is the standard boilerplate that calls the main() function.
if __name__ == '__main__':
    main()