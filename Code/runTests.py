import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import dictionaryAttack as attack
import bagging as bag
from sklearn.metrics import roc_curve, auc
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression

def computeError(predictions, y_test):
    return np.mean(predictions != y_test)

# Main function to run the dictionary attack
def main():
    # TASK 0: SET TEST PARAMETERS
    frac_knowl = 1 # optimal dicionary attack
    frac_mal_instances = .3
    no_predictors = 100
    no_iterations = 1

    
    # TASK 1: Read binary feature set from 'Features.csv' and 'Labels.csv'
    df_X = pd.read_csv('Features.csv', header = None)
    X = np.array(df_X)
    #print X
    
    df_y = pd.read_csv('Labels.csv', header = None)
    y = np.array(df_y).T[0]
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
    base_classifier = LogisticRegression()
    base_classifier.fit(X_train, y_train)
    predictions = base_classifier.predict(X_test)
    base_error = computeError(predictions, y_test)
    #print predictions
    print 'base linear classifier on clean data:', base_error
    
    
    # TASK 5: Train a base linear classifier on poisoned data
    base_classifier_poisoned = LogisticRegression()
    base_classifier_poisoned.fit(X_train_poisoned, y_train_poisoned)
    predictions = base_classifier_poisoned.predict(X_test)
    base_error_poisoned = computeError(predictions, y_test)
    #print predictions
    print 'base linear classifier on poisoned data:', base_error_poisoned   
    
    
    # TASK 6: Train a bagged linear classifier on clean data
    bag_clean_errors = bag.bagPredictors(X_train, y_train, X_test, y_test, no_predictors)
    
    #TASK 7: Train a bagged linear classifier on poisoned data
    bag_poisoned_errors = bag.bagPredictors(X_train_poisoned, y_train_poisoned, X_test, y_test, no_predictors)
    
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
    plt.savefig('LogisticRegresstion.png', bbox_inches='tight')


# This is the standard boilerplate that calls the main() function.
if __name__ == '__main__':
    main()