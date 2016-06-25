
# coding: utf-8

import numpy as np
import pandas as pd

# Definition of the sigmoid function
def sigmoid(z):
    prediction = 1.0/(1.0 + np.exp(-z))
    
    return prediction
    
# Hypothesis function
def hypothesis(theta, x):
    z = np.dot(theta,x)
    prediction = sigmoid(z)
    
    return prediction

	

# Cost for a given training example
def costFunction(prediction, trueValue):
    cost = trueValue * np.log(prediction) + (1 - trueValue) * np.log(1 - prediction)
    
    print 'Prediction:', prediction, 'Class:', trueValue
    print 'Cost of prediction:', cost
	
    return cost


# Total cost over n training instances; This is the function we are minimizing
def computeCost(predictions, trueValues):
    n_instances = len(predictions)
    cost_sum = 0
    
    for instance in range(0, n_instances):
        cost_sum -= costFunction(predictions[instance], trueValues[instance])
    
    
    print 'Sum cost is', cost_sum
    
    cost = (1.0/n_instances)*cost_sum
    print 'Cost is ', cost
    
    return cost
    

    
# Cost for a given training example
def gradient(x, trueValue, theta, j, m):
    grad = (hypothesis(theta,x) - trueValue) * x[j]
    print 'TODO'


# INPUT: X := 2D 
def regLogisticRegression():
    print 'TODO'
    

# Main function to write tests
def main():

    df_X = pd.read_csv('BinaryFeatures.csv', header = None)
    print df_X
    
    X = np.array(df_X) 
    print X
    
    df_y = pd.read_csv('Labels.csv', header = None)
    print df_y
    
    y = np.array(df_y)
    print y

    regLogisticRegression()
    
    prediction = sigmoid(2)
    print prediction
    
    cost = costFunction(prediction, 0)
    print cost
    
    computeCost([.98,.002,.89],[1,0,1])


# Standard boilerplate to call the main() function to begin
# the program.
if __name__ == '__main__':
  main()


# In[ ]:



