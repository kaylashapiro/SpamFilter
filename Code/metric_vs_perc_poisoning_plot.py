# coding: utf-8

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from bagging_plot import get_results_path, get_classifier_name


def metric_vs_perc_poisoning(classifier, dataset, attack, 
                             percent_poisonings, 
                             num_baggers, 
                             metric,
                             include_baggers=False,
                             ):
    metrics = {
        'Error Rate': 0,
        'TPR': 1,
        'FPR': 2,
        'FNR': 3,
        'TNR': 4,
        'AUC': 5,
    }
    
    y_axis_metric = metrics[metric]

    classifier_name = get_classifier_name(classifier)
    attack_name = get_attack_name(attack)
    
    num_lines_to_plot = len(num_baggers)
    
    X = np.hstack(([0], percent_poisonings))
    
    base_data = load_base_data(classifier, dataset, attack, y_axis_metric, percent_poisonings)

    plt.plot(X, base_data, label='base classifier')
    
    if include_baggers:
        data = load_bagging_data(classifier, dataset, attack, y_axis_metric, percent_poisonings, num_baggers)
              
        for line in xrange(num_lines_to_plot):
            plt.plot(X, data[line], label=get_label(num_baggers, line))
        if y_axis_metric == 0:
            legend = plt.legend(loc='upper left', shadow=True, prop={'size':12})
    
    title = classifier_name + ' (' + get_attack_name(attack) + ')'
    
    plt.xlabel('Percent Poisoning', fontsize=16)
    plt.ylabel(metric, fontsize=16)
    #plt.title(title, fontsize=20)
    
    return plt
    
def load_bagging_data(classifier, dataset, attack, y_axis_metric,
              ## params
              percent_poisonings=[10, 20, 30], 
              num_baggers=[3, 5, 10, 20, 50]
              ):
    '''
    Input:
    - classifier: string; ['adaline', 'logistic_regression', 'naivebayes']
    - dataset: string; ['enron']
    - attack: string; ['No', 'Dict', 'Empty', 'Ham']
    - percent_poisonings: int array; [10, 20, 30]
    - num_baggers: int array; bagging slices to take
    
    Output:
    - data: M * N Numpy matrix
            with M: number of bagging slices
            and  N: number of poisoning slices
    '''
    
    num_lines_to_plot = len(num_baggers)
    
    num_baggers = num_baggers - np.ones(num_lines_to_plot, dtype='int8')
        
    base_path = get_results_path(classifier, dataset, 'No')
    bagging_file = '100_100_0.csv'
    
    df_base_bagger = pd.read_csv(base_path + bagging_file, header = 0)
    base_bagger = np.array(df_base_bagger)
    y_axis_metrics = base_bagger[:,y_axis_metric]
    
    data = np.reshape(y_axis_metrics[num_baggers], (num_lines_to_plot, 1))
    
    for percent_poison in percent_poisonings:
        path = get_results_path(classifier, dataset, attack, percent_poison)
        df_poison_bagger = pd.read_csv(path + bagging_file, header = 0)
        bagger = np.array(df_poison_bagger)
        y_axis_metrics = bagger[:,y_axis_metric]
        bagger_data = np.reshape(y_axis_metrics[num_baggers], (num_lines_to_plot, 1))
        data = np.hstack((data, bagger_data))
    
    return data
    
def load_base_data(classifier, dataset, attack, y_axis_metric,
                   ## params
                   percent_poisonings=[10, 20, 30]
                   ):
    base_path = get_results_path(classifier, dataset, 'No')
    base_file = '0_0_0.csv'
    df_base = pd.read_csv(base_path + base_file, header = 0)
    base = np.array(df_base)
    
    data = [base[0][y_axis_metric]] 
    
    for percent_poison in percent_poisonings:
        path = get_results_path(classifier, dataset, attack, percent_poison)
        df_poison_base = pd.read_csv(path + base_file, header = 0)
        base = np.array(df_poison_base)
        data.append(base[0][y_axis_metric])
                
    return data
    
def get_label(num_baggers, line):
    label = str(num_baggers[line]) + ' baggers'
    
    return label
    
def get_attack_name(attack, percent_poisoning=None):
    attacks = {
        'Dict': 'Dictionary Attack',
        'Empty': 'Empty Attack',
        'Ham': 'Ham Attack',
    }
    
    attack_name = attacks[attack]
    
    return attack_name    
    
def get_filename(classifier,
                 dataset,
                 attack,
                 metric,
                 include_baggers):
    classifier_names = {
        'adaline' : 'AD_',
        'logistic_regression' : 'LR_',
        'naivebayes' : 'NB_',
    }
    
    classifier_name = classifier_names[classifier]
    
    base = '_'
    
    if not include_baggers:
        base = '_base_'
    
    if metric is 'Error Rate':
        metric = 'error'
    
    name = classifier_name + dataset + base + attack + '_' + metric + '_vs_perc_poisoning.png'
    
    return name   

def main():
    classifier = 'logistic_regression'
    #dataset = 'enron'
    #attack = 'Ham'
    
    
    #metric = 'FPR'
    
    
    datasets = ['enron', 'lingspam']
    attacks = ['Dict', 'Empty', 'Ham']
    percent_poisoning = [10, 20, 30]
    num_baggers = [3, 5, 10, 20, 50]
    metrics = ['Error Rate', 'TPR', 'FPR', 'FNR', 'TNR', 'AUC']
    include_baggers=True
    
    for dataset in datasets:
        for attack in attacks:
            for metric in metrics:
                plot = metric_vs_perc_poisoning(classifier, 
                                                dataset, 
                                                attack, 
                                                percent_poisoning, 
                                                num_baggers, 
                                                metric,
                                                include_baggers)

                name = get_filename(classifier, dataset, attack, metric, include_baggers)                                
                print name
                path = '../Plots/bagging/'
                plot.savefig(path + name)
                plot.gcf().clear()
    
    return
    
if __name__ == '__main__':
    main()