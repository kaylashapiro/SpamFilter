# coding: utf-8

import sys
import importlib
import pandas as pd
import numpy as np

sys.path.insert(0, 'attacks')

def performAttack(frac_knowl=1, 
                  perc_poisoning=[10, 20, 30], 
                  no_iterations=10,
                  attack=None,
                  attack_folder=None,
                  dataset=None):
    '''
    Performs attacks on no_iterations training sets at various poisoning levels
    and saves them to .csv files.
    
    Inputs (default):
    - frac_knowl: default perfect knowledge; percentage of features the attacker knows
                  float between 0 and 1
    - perc_poisoning: default list of real numbers ranging from 0 to 100
    - no_iterations: Integer number of training sets to iterate through and attacks
                     to generate at each perc_poisoning
    - attack: string; choose from 1) 'dictionaryAttack' 2) 'emptyAttack'
    - attack_folder: string; choose from 1) 'DictAttackData' 2) 'EmptyAttackData'
    - dataset: string; choose from 1) 'enron' 2) 'lingspam'
    
    Output:
    NONE
    '''
    try:
        attack = importlib.import_module(attack)
    except ImportError as error:
        print error
        print "Failed to import attack module in performAttacks.py"
        print "Available attack modules: 1) 'dictionaryAttack' 2) 'emptyAttack'"
        sys.exit(0)
        
    path = '../Datasets/' + attack_folder + '/' + dataset + '/'
    path_train = '../Datasets/TrainData/' + dataset + '/'
    
    for iter in xrange(no_iterations):    
        X_train_name = 'X_train_' + str(iter) + '.csv'
        y_train_name = 'y_train_' + str(iter) + '.csv'
        
        df_X_train = pd.read_csv(path_train + X_train_name, header = None)
        X_train = np.array(df_X_train)
        
        df_y_train = pd.read_csv(path_train + y_train_name, header = None)
        y_train = np.array(df_y_train)
        
        for perc in perc_poisoning:
            print 'ATTACK', iter
            print 'POISONING', perc
            folder = str(perc) + '_perc_poison/'
            perc = float(perc)/100
            
            [X_train_poisoned, y_train_poisoned] = attack.poisonData(X_train, y_train, perc, frac_knowl)
            
            X_name = 'X_train_' + str(iter) + '.csv'
            y_name = 'y_train_' + str(iter) + '.csv'
        
            X_path = path + folder + X_name
            y_path = path + folder + y_name
        
            my_X_train = pd.DataFrame(X_train_poisoned, dtype = np.uint8)
            my_X_train.to_csv(X_path, index=False, header=False)
    
            my_y_train = pd.DataFrame(y_train_poisoned, dtype = np.uint8)
            my_y_train.to_csv(y_path, index=False, header=False)
    

    
# Main function to run algorithm on various fractions of attacker knowledge and control.
def main():    
    performAttack(attack='empty',attack_folder='EmptyAttackData',dataset='lingspam')
    
    
# This is the standard boilerplate that calls the main() function.
if __name__ == '__main__':
    main()