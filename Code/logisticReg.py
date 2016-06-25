
# coding: utf-8

# In[18]:

import numpy as np


# In[19]:

def sigmoid(z):
    prediction = 1.0/(1.0 + np.exp(-z))
    
    return prediction
    


# In[20]:

def hypothesis(theta, x):
    z = np.dot(theta,x)
    prediction = sigmoid(z)
    
    return prediction


# In[21]:

# Cost for a given training example

def costFunction(prediction, trueValue):
    cost = trueValue * np.log(prediction) + (1 - trueValue) * np.log(1 - prediction)
    
    return cost


# In[50]:

# Total cost over n training instances

def totalCost(predictions, trueValues):
    n_instances = len(predictions)
    cost_sum = 0
    
    for instance in range(0, n_instances):
        cost_sum += costFunction(predictions[instance], trueValues[instance])
        print cost_sum


# In[60]:

# INPUT: X := 2D 

def regLogisticRegression():
    print 'TODO'
    


# In[61]:

# Main function to write tests
def main():
    regLogisticRegression()
    
    prediction = sigmoid(2)
    print prediction
    
    cost = costFunction(prediction, 0)
    print cost
    
    totalCost([.98,.002,.89],[1,0,1])


# In[62]:

# Standard boilerplate to call the main() function to begin
# the program.
if __name__ == '__main__':
  main()


# In[ ]:



