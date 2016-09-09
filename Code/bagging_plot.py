# coding: utf-8

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def bag_vs_base_plot(classifier, dataset, attack, percent_poisoning, metric):
    '''
    Method to plot y-axis metric vs x-axis number of baggers.
    Plots clean and poisoned data
    '''
    metrics = {
        'Error Rate': 0,
        'TPR': 1,
        'FPR': 2,
        'FNR': 3,
        'TNR': 4,
        'AUC': 5,
    }
    
    y_axis_metric = metrics[metric]

    no_attack_path = get_results_path(classifier, dataset, 'No')
    attack_path = get_results_path(classifier, dataset, attack, percent_poisoning)
    
    base_file = '0_0_0.csv'
    bagging_file = '100_100_0.csv'
    
    no_attack_base = no_attack_path + base_file
    no_attack_bag = no_attack_path + bagging_file
    attack_base = attack_path + base_file
    attack_bag = attack_path + bagging_file
    
    df_no_attack_base = pd.read_csv(no_attack_base, header = 0)
    no_attack_base = np.array(df_no_attack_base)
    
    df_no_attack_bag = pd.read_csv(no_attack_bag, header = 0)
    no_attack_bag = np.array(df_no_attack_bag)
    
    df_attack_base = pd.read_csv(attack_base, header = 0)
    attack_base = np.array(df_attack_base)
    
    df_attack_bag = pd.read_csv(attack_bag, header = 0)
    attack_bag = np.array(df_attack_bag)
    
    N = attack_bag.shape[0]
    
    plot = metric_vs_num_baggers(classifier, attack, percent_poisoning,
                                no_attack_base[0][y_axis_metric], no_attack_bag[:,y_axis_metric], attack_base[0][y_axis_metric], attack_bag[:,y_axis_metric], N, metric)
    
    return plot
    
    
def metric_vs_num_baggers(classifier, attack, percent_poisoning,
                         no_attack_base_error, 
                         no_attack_bag_errors, 
                         attack_base_error,
                         attack_bag_errors,
                         N,
                         metric,
                         ):    
    no_attack_base_errors = np.repeat(no_attack_base_error, N)
    attack_base_errors = np.repeat(attack_base_error, N)
    
    X = np.linspace(1, N, num=N, endpoint=True)
    
    title = get_attack_name(attack, percent_poisoning)
    
    plt.title(title, fontsize=18)
    
    plt.xlabel('Number of Baggers')
    plt.ylabel(metric)
    
    no_attack_base = plt.plot(X, no_attack_base_errors, 'b--', 
                              label=get_classifier_name(classifier))
    no_attack_bag = plt.plot(X, no_attack_bag_errors, 'b',
                             label='Bagged')
    attack_base = plt.plot(X, attack_base_errors, 'r--',
                           label=get_classifier_name(classifier, percent_poisoning))
    attack_bag = plt.plot(X, attack_bag_errors, 'r',
                          label='Bagged (poisoned)')
    
    #legend = plt.legend(loc='upper right', shadow=True, prop={'size':12})
    
    return plt
    
def get_attack_name(attack, percent_poisoning=None):
    attacks = {
        'Dict': 'Dictionary Attack',
        'Empty': 'Empty Attack',
        'Ham': 'Ham Attack',
    }
    
    attack_name = attacks[attack]
    
    if percent_poisoning is not None:
        poisoning_name = ' (' + str(percent_poisoning) + ' percent poisoning)'
    
    return str(percent_poisoning) + ' percent poisoning'#attack_name + poisoning_name
    
def get_classifier_name(classifier, percent_poisoning=None):
    classifiers = {
        'adaline': 'Adaline',
        'logistic_regression': 'Logistic Regression',
        'naivebayes': 'Naive Bayes',
        'boldAdaline': 'Bold Adaline',
        'adaline_with_adagrad': 'Adaline with AdaGrad',
        'adaline_with_adadelta': 'Adaline with AdaDelta',
    }
    
    classifier_name = classifiers[classifier]
    
    if percent_poisoning is not None:
        classifier_name = classifier_name + ' (poisoned)'
        
    return classifier_name
    
def get_results_path(classifier,
             dataset,
             attack,
             percent_poisoning=None):
    results_folder = '../Results/'
    results_path = results_folder + classifier + '/' + dataset + '/'
    
    if percent_poisoning is None:
        data_folder = ''
    else:
        data_folder = str(percent_poisoning) + '_perc_poison/'
    
    attack_path = results_path + attack + 'Attack/' + data_folder
    
    return attack_path
    
def main():
    classifier = 'logistic_regression'
    dataset = 'enron'
    attack = 'Dict'
    percent_poisoning = 10
    metric = 'Error Rate'
    
    plot = bag_vs_base_plot(classifier, dataset, attack, percent_poisoning, metric)
    plot.show()
    
    
if __name__ == '__main__':
    main()