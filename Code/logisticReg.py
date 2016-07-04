
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

	
def computePredictions(theta, X, n_instances):
    predictions = []
    
    for i in range(0, n_instances):
        predictions.append(hypothesis(theta, X[i]))
        
    return predictions

    
# Cost for a given training example
def costFunction(prediction, trueValue):
    cost = trueValue * np.log(prediction) + (1 - trueValue) * np.log(1 - prediction)
    
    #print 'Prediction:', prediction, 'Class:', trueValue
    #print 'Cost of prediction:', cost
	
    return cost


# Total cost over n training instances; This is the function we are minimizing
def computeCost(predictions, trueValues, n_instances):
    cost_sum = 0
    
    for instance in range(0, n_instances):
        cost_sum -= costFunction(predictions[instance], trueValues[instance])
    
    
    #print 'Sum cost is', cost_sum
    
    cost = (1.0/n_instances)*cost_sum
    print 'Cost:', cost
    
    return cost
    

    
# Gradient for a given feature
def costGradient(X, y, theta, j, n_instances): 
    grad_sum = 0
    
    for i in range(0, n_instances):
        x_i = X[i]
        x_ij = x_i[j]
        
        y_i = y[i]
        
        h = hypothesis(theta, x_i)
        
        grad_sum += (h - y_i) * x_ij
        
    grad = (1.0/float(n_instances)) * grad_sum
    
    #print 'The gradient is', grad
    return grad
    

# Gradient descent
def gradientDescent(X, y, theta, n_instances, n_features, alpha):
    new_theta = []
    
    for j in range(n_features):
        new_thetaj = theta[j] - alpha * costGradient(X,y,theta,j,n_instances)
        new_theta.append(new_thetaj)
        
    return new_theta

# X holds training instances, y holds their class values
def regLogisticRegression(X, y):
    n_instances, n_features = X.shape
    
    theta = [0] * n_features
    alpha = 1
    n_iters = 1
    
    for x in range(0,n_iters):
        theta = gradientDescent(X,y,theta,n_instances,n_features,alpha)
        
        if x%1 == 0:
            predictions = computePredictions(theta, X, n_instances)
            #print 'Iteration:', x
            #print 'Theta:', theta
            computeCost(predictions, y, n_instances)

    
    return theta
    
    

# Main function to write tests
def main():

    df_X = pd.read_csv('Features.csv', header = None)
    #print df_X
    
    X = np.array(df_X) 
    print X
    
    df_y = pd.read_csv('Labels.csv', header = None)
    #print df_y
    
    y = np.array(df_y).T[0]
    print y
    
    #n_instances, n_features = X.shape
    #print n_instances, n_features

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




