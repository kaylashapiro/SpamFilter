# This is a program to create "bootstrap" replicate sets from a single dataset.
# This idea equates to sampling with replacement.

import numpy as np 
import random
import pandas as pd
import sys
    
# Generate a single bootstrap replicate set by sampling with replacement.    
def generateReplicate(X, y, new_instances):
    indices = np.random.choice(X.shape[0], new_instances, replace=True)
     
    return (np.array(X[indices]), y[indices])
    

# Function to generate a certain number of replicate sets.
# Saves as csv for specified path
def generateBootstraps(X, y, new_instances, n_replicates):
    folder = './Bootstraps/'

    for i in range(0, n_replicates):
        replicate, labels = generateReplicate(X, y, new_instances)
        
        filename = 'replicate' + str(i+1) + '.csv'
        out_name = folder + filename
        
        with open(out_name, 'wb') as ofile:
            np.savetxt(ofile, replicate, fmt='%u', delimiter=',')
        
        
# Main function to create bootstrap replicate sets.
# argv[1] := number of replicates to generate.
def main():
    if len(sys.argv) >= 2:
        n_replicates = int(sys.argv[1])
    else:
        n_replicates = 25
  
    new_instances = 5
  
    # Let's get a dataset up in this bizniz
    df_X = pd.read_csv('test.csv', header = None)
    X = np.array(df_X)
    print X
    
    df_y = pd.read_csv('test_y.csv', header = None)
    y = np.array(df_y)
    print y
  
    [X, y] = generateReplicate(X, y, X.shape[0])
    print X
    print y

# This is the standard boilerplate that calls the main() function.
if __name__ == '__main__':
    main()