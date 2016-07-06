import numpy as np
import random
import pandas as pd
import logisticRegVec as reg

# Function to generate dictionary attack data
# INPUT: number of poison data instances, number of features for each data point, number of features the attacker knows
def generateAttackData(no_mal_instances, no_features, no_mal_features):
    rand_features = np.array([0] * (no_features - no_mal_features) + [1] * no_mal_features)
    np.random.shuffle(rand_features)
    
    mal_data = np.array([rand_features,] * no_mal_instances)
    mal_y = np.array([1] * no_mal_instances) # Contamination assumption
      
    return (mal_data, mal_y)
    
# Function to perform dictionary attack    
# INPUT: Data, labels, features attacker knows (as a fraction), poisoned data (as a fraction).
def dictionaryAttack(X, y, frac_knowl, frac_mal_instances):
    no_instances, no_features = X.shape
    
    no_mal_features = int(round(frac_knowl * no_features))
    no_mal_instances = int(round(frac_mal_instances * (no_instances / (1 - frac_mal_instances))))
        
    mal_data, mal_y = generateAttackData(no_mal_instances, no_features, no_mal_features)
    
    X_train = np.concatenate((X, mal_data), 0)
    y_train = np.append(y, mal_y)
    
    #reg.regLogisticRegression(X, y)
    


# Main function to run the dictionary attack
def main():
    df_X = pd.read_csv('test.csv', header = None)
    X = np.array(df_X)
    print X
    
    df_y = pd.read_csv('test_y.csv', header = None)
    y = np.array(df_y).T[0]
    print y
    
    frac_knowl = .5
    frac_mal_instances = 1.0/3
    
    dictionaryAttack(X, y, frac_knowl, frac_mal_instances)
    
    

# This is the standard boilerplate that calls the main() function.
if __name__ == '__main__':
    main()