# This is a program to create "bootstrap" replicate sets from a single dataset.
# This idea equates to sampling with replacement.

import numpy as np 
#import pandas as pd
import sys

# n_instances := number of instances to have in the replicate set.
# Returns an array holding (random  indices of data instances to put into the set.
def sampleWithReplacement(n_instances):

# Main function to create bootstrap replicate sets
# argv[1] := number of replicates to generate
def main():
  if len(sys.argv) >= 2:
	n_replicates = sys.argv[1]
  else:
    n_replicates = 25
  print n_replicates

# This is the standard boilerplate that calls the main() function.
if __name__ == '__main__':
  main()