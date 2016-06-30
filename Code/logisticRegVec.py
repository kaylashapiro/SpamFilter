
# coding: utf-8

import numpy as np
import pandas as pd

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
    
    
    print 'Cost:', cost
    
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
    alpha = 1
    n_iters = 10000
    
    for x in range(0,n_iters):
        theta = gradientDescent(X,y,theta,n_instances,n_features,alpha)
        
        if x%100 == 0:
            print 'Iteration:', x
            #print 'Theta:', theta
            computeCost(X, theta, y, n_instances)

    
    return theta
    
    

# Main function to write tests
def main():

    df_X = pd.read_csv('BinaryFeatures.csv', header = None)
    #print df_X
    
    X = np.array(df_X) 
    print X
    
    X = np.array(X)
    print X
    
    df_y = pd.read_csv('Labels.csv', header = None)
    #print df_y
    
    y = np.array(df_y).T[0]
    print y
    
    
    theta = regLogisticRegression(X,y)    
    print 'Theta:', theta
    
    #prediction = sigmoid(2)
    #print prediction
    
    #cost = costFunction(prediction, 0)
    #print cost
    
    #computeCost([.98,.002,.89],[1,0,1], 2)
    

# Standard boilerplate to call the main() function to begin
# the program.
if __name__ == '__main__':
  main()