
# coding: utf-8

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Function to account for bias term
def addBias(X):
    n_instances = X.shape[0]   
    bias = np.ones(n_instances).T
    
    return np.insert(X, 0, bias, axis=1)
    

# Definition of the sigmoid function
def sigmoid(z):
    prediction = np.divide(1.0, (1.0 + np.exp(np.multiply(-1.0,z))))
    
    return prediction
    
    
# Hypothesis function
def hypothesis(X, theta):
    z = np.dot(X, theta)
    predictions = sigmoid(z)
    
    return predictions


# Total cost over n training instances; This is the function we are minimizing
def computeCost(X, theta, trueValues, n_instances):
    predictions = hypothesis(X, theta)
    
    #print 'Predictions:', predictions

    ones = np.ones(n_instances)
    
    cost_sum = np.sum(trueValues * np.log(predictions) + (ones - trueValues) * np.log(ones - predictions))
    
    cost = -(1.0/n_instances)*cost_sum
      
    #print 'Cost:', cost
    
    return cost
    

# Gradient descent
def gradientDescent(X, y, theta, n_instances, n_features, alpha):
    predictions = hypothesis(X, theta)

    grad = np.dot(X.transpose(), np.divide((y - predictions),n_instances))
    
    new_theta = theta + np.multiply(grad, alpha)
     
    return new_theta

    
# X holds training instances, y holds their class values
def regLogisticRegression(X, y):
    n_instances, n_features = X.shape
    
    theta = [0] * n_features
    cost = []
    alpha = 1
    n_iters = 1000
    
    for x in range(0,n_iters):
        theta = gradientDescent(X,y,theta,n_instances,n_features,alpha)
        cost.append(computeCost(X, theta, y, n_instances))
        
        if x%100 == 0:
            print 'Iteration:', x
            #print 'Theta:', theta
            print 'Cost:', cost[x]
    
    plotCost(n_iters, cost)
            
    return theta
    

# Plot (iteration, cost at iteration)
def plotCost(n_iters, cost):
    x_axis = np.linspace(1,n_iters, num=n_iters, endpoint=True)
    
    plt.plot(x_axis,cost)
    plt.xlabel('Iterations')
    plt.ylabel('Cost')
    plt.show()
    
    
# Main function to write tests and run the regression
def main():

    df_X = pd.read_csv('Features.csv', header = None)
    #print df_X
    
    X = np.array(df_X) 
    print X
    
    df_y = pd.read_csv('Labels.csv', header = None)
    #print df_y
    
    y = np.array(df_y).T[0]
    print y
    
    X = addBias(X)
    
    print X
    
    theta = regLogisticRegression(X,y)    
    print 'Theta:', theta


# Standard boilerplate to call the main() function to begin
# the program.
if __name__ == '__main__':
  main()