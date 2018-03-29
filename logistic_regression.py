import numpy as np







def sigmoid(z):
    """Calculate the sigmoid of all elements of a np array"""
    
    g = 1 / (1 + np.exp(-z))
    
    return(g)



def cost_function(theta, X, y):
    """Calculate the cost for given theta, X and y"""
    
    m, n = X.shape
    
    theta = theta.reshape((n, 1))
        
    x_dot_theta = X.dot(theta)

    J = np.sum((-y * np.log(sigmoid(x_dot_theta))) - ((1 - y) * np.log(1 - sigmoid(x_dot_theta)))) / m
    
    return(J)


def gradient(theta, X, y):
    """Calculate the gradient of the cost function w.r.t. each element of theta"""
    
    m, n = X.shape
    
    theta = theta.reshape((n, 1))
    
    grad = (X.T).dot(sigmoid(X.dot(theta)) - y) / m
    
    return(grad.flatten())



