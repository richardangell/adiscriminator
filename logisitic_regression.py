import numpy as np








def sigmoid(z):
    """Calculate the sigmoid of all elements of a np array"""

    g = 1 / (1 + np.exp(-z))

    return(g)



def cost_function(theta, X, y):
    """Calculate the cost for given theta, X and y"""

    # number of training examples
    m = y.shape[0]

    # value of the cost function
    J = sum((-y .* np.log(sigmoid( X * theta))) - ((1 - y) .* log(1 - sigmoid( X * theta)))) / m
    
    return(J)


def gradient(theta, X, y):
    """Calculate the gradient of the cost function w.r.t. each element of theta"""







