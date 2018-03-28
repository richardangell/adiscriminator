import numpy as np







def sigmoid(z):
    """Calculate the sigmoid of all elements of a np array"""
    
    g = 1 / (1 + np.exp(-z))
    
    return(g)



def cost_function(theta, X, y):
    """Calculate the cost for given theta, X and y"""
    
    m = y.shape[0]
    
    x_dot_theta = X.dot(theta)
    
    # value of the cost function
    J = sum((-y * np.log(sigmoid(x_dot_theta))) - ((1 - y) * np.log(1 - sigmoid(x_dot_theta)))) / m
    
    return(J)


def gradient(theta, X, y):
    """Calculate the gradient of the cost function w.r.t. each element of theta"""
    
    m, n = y.shape
    
    grad = (X.T).dot(sigmoid(X.dot(theta)) - y)
    
    return(grad)




Result = op.minimize(fun = cost_function, 
                    x0 = initial_theta, 
                    args = (X, y),
                    method = 'TNC',
                    jac = gradient)
