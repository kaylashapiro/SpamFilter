import numpy as np
import random
import pandas as pd
import logisticRegVec as reg

# Function to return a random index of a ham email
# INPUT: Class label array (consisting of 1s and 0s)
def selectHamIndex(y):
    ham_index = [i for i in range(len(y)) if y[i] == 0]
    
    return np.random.choice(ham_index, 1)

    
# Function to generate dictionary attack data
# INPUT: number of poison data instances, number of features for each data point, number of features the attacker knows
def generateAttackData(ham_email, features_present, no_mal_instances, no_features, no_mal_features):   
    rand_features = np.array([0] * (no_features - no_mal_features) + [1] * no_mal_features)
    np.random.shuffle(rand_features)
    
    mal_ham = np.array(ham_email)
    
    for i in range(len(rand_features)):
        mal_ham[features_present[i]] = rand_features[i]
  
    mal_data = np.array([mal_ham,] * no_mal_instances)
    mal_y = np.array([1] * no_mal_instances) # Contamination assumption
      
    return (mal_data, mal_y)
    
    
# Function to perform dictionary attack    
# INPUT: Data, labels, features attacker knows (as a fraction), poisoned data (as a fraction).
def focusedAttack(X, y, frac_knowl, frac_mal_instances):
    ham_email = X[selectHamIndex(y)][0]
    
    features_present = [i for i in range(len(ham_email)) if ham_email[i] == 1]
    
    no_features = len(features_present)

    no_instances = X.shape[0]
    
    no_mal_features = int(round(frac_knowl * no_features))
    no_mal_instances = int(round(frac_mal_instances * (no_instances / (1 - frac_mal_instances))))
        
    mal_data, mal_y = generateAttackData(ham_email, features_present, no_mal_instances, no_features, no_mal_features)
    
    X_train = np.concatenate((X, mal_data), 0)
    y_train = np.append(y, mal_y)
    
    print X_train
    print y_train
    
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
    
    focusedAttack(X, y, frac_knowl, frac_mal_instances)
    

# This is the standard boilerplate that calls the main() function.
if __name__ == '__main__':
    main()