# coding: utf-8

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from bagging_plot import get_results_path, get_classifier_name

'''
Methods to plot adaptive learning rate results.
'''


def metric_vs_perc_poisoning(dataset, attack, 
                             percent_poisonings, 
                             metric,
                             ):
    '''
    Plots each of the adaptive learning rate methods for a given attack and dataset.
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

    attack_name = get_attack_name(attack)
        
    X = np.hstack(([0], percent_poisonings))
    
    adaline_data = load_base_data('adaline', dataset, attack, y_axis_metric, percent_poisonings)
    plt.plot(X, adaline_data, label='Adaline')
    
    boldAdaline_data = load_base_data('boldAdaline', dataset, attack, y_axis_metric, percent_poisonings)
    plt.plot(X, boldAdaline_data, label='Bold Adaline')
 
    adaline_with_adagrad_data = load_base_data('adaline_with_adagrad', dataset, attack, y_axis_metric, percent_poisonings)
    plt.plot(X, adaline_with_adagrad_data, label='Adaline with AdaGrad')
    
    adaline_with_adadelta_data = load_base_data('adaline_with_adadelta', dataset, attack, y_axis_metric, percent_poisonings)
    plt.plot(X, adaline_with_adadelta_data, label='Adaline with AdaDelta')
    
    legend = plt.legend(loc='upper left', shadow=True, prop={'size':12})
    
    title = 'Adaline Adaptive Learning Rate Methods' + ' (' + get_attack_name(attack) + ')'
    
    plt.xlabel('Percent Poisoning', fontsize=16)
    plt.ylabel(metric, fontsize=16)
    
    return plt
    
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
    
def get_attack_name(attack, percent_poisoning=None):
    attacks = {
        'Dict': 'Dictionary Attack',
        'Empty': 'Empty Attack',
        'Ham': 'Ham Attack',
        'Ham2': 'MI Ham Attack',
    }
    
    attack_name = attacks[attack]
    
    return attack_name    
    
def get_filename(dataset,
                 attack,
                 metric):
    
    classifier_name = 'ALRs_'
    
    base = '_'
    
    if metric is 'Error Rate':
        metric = 'error'
    
    name = classifier_name + dataset + base + attack + '_' + metric + '_vs_perc_poisoning.png'
    
    return name   

def main():
    dataset = 'enron'
    attack = 'Empty'    
    
    percent_poisoning = [10, 20, 30]
    metrics = ['Error Rate', 'FPR', 'FNR', 'AUC']
    
    for metric in metrics:
        plot = metric_vs_perc_poisoning(dataset, 
                                        attack, 
                                        percent_poisoning, 
                                        metric)

        name = get_filename(dataset, attack, metric)                                
        
        path = '../Plots/adaptive_learning_rate/'
        
        plot.savefig(path + name)
        plot.gcf().clear()
    
    return
    
if __name__ == '__main__':
    main()