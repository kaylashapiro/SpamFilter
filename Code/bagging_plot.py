# coding: utf-8

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def bag_vs_base_plot(classifier, dataset, attack, percent_poisoning):
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
    no_attack_base_error = no_attack_base[0][0]
    
    df_no_attack_bag = pd.read_csv(no_attack_bag, header = 0)
    no_attack_bag = np.array(df_no_attack_bag)
    no_attack_bag_errors = no_attack_bag[:,0]
    
    df_attack_base = pd.read_csv(attack_base, header = 0)
    attack_base = np.array(df_attack_base)
    #print attack_base
    attack_base_error = attack_base[0][0]
    #print attack_base_error
    
    df_attack_bag = pd.read_csv(attack_bag, header = 0)
    attack_bag = np.array(df_attack_bag)
    #print attack_bag
    attack_bag_errors = attack_bag[:,0]
    #print attack_bag_error
    
    N = attack_bag.shape[0]
    
    error_plot = bag_vs_base_plot_error(no_attack_base[0][0], no_attack_bag[:,0], attack_base[0][0], attack_bag[:,0], N)

def bag_vs_base_plot_error(classifier, dataset, attack, percent_poisoning):
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
    no_attack_base_error = no_attack_base[0][0]
    
    df_no_attack_bag = pd.read_csv(no_attack_bag, header = 0)
    no_attack_bag = np.array(df_no_attack_bag)
    no_attack_bag_errors = no_attack_bag[:,0]
    
    df_attack_base = pd.read_csv(attack_base, header = 0)
    attack_base = np.array(df_attack_base)
    #print attack_base
    attack_base_error = attack_base[0][0]
    #print attack_base_error
    
    df_attack_bag = pd.read_csv(attack_bag, header = 0)
    attack_bag = np.array(df_attack_bag)
    #print attack_bag
    attack_bag_errors = attack_bag[:,0]
    #print attack_bag_error
    
    N = attack_bag.shape[0]
    
    no_attack_base_errors = np.repeat(no_attack_base_error, N)
    attack_base_errors = np.repeat(attack_base_error, N)
    
    print no_attack_base_errors
    print no_attack_bag_errors
    print attack_base_errors
    print attack_bag_errors
    
    X = np.linspace(1, N, num=N, endpoint=True)
    print X
    
    title = get_attack_name(attack, percent_poisoning)
    
    plt.title(title, fontsize=18)
    
    #plt.xlabel('Number of Baggers')
    #plt.ylabel('Error Rate')
    
    no_attack_base = plt.plot(X, no_attack_base_errors, 'b--', 
                              label=get_classifier_name(classifier))
    no_attack_bag = plt.plot(X, no_attack_bag_errors, 'b',
                             label='Bagged')
    attack_base = plt.plot(X, attack_base_errors, 'r--',
                           label=get_classifier_name(classifier, percent_poisoning))
    attack_bag = plt.plot(X, attack_bag_errors, 'r',
                          label='Bagged (poisoned)')
    
    legend = plt.legend(loc='upper right', shadow=True, prop={'size':12})
    plt.show()
    
def get_attack_name(attack, percent_poisoning):
    attacks = {
        'Dict': 'Dictionary Attack',
        'Empty': 'Empty Attack',
        'Ham': 'Ham Attack',
    }
    
    attack_name = attacks[attack]
    poisoning_name = ' (' + str(percent_poisoning) + ' percent poisoning)'
    
    return str(percent_poisoning) + ' percent poisoning'#attack_name + poisoning_name
    
def get_classifier_name(classifier, percent_poisoning=None):
    classifiers = {
        'adaline': 'Adaline',
        'logistic_regression': 'Logistic Regression',
        'naivebayes': 'Naive Bayes',
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
    
    bag_vs_base_plot_error(classifier, dataset, attack, percent_poisoning)

if __name__ == '__main__':
    main()