import numpy as np
import scipy.optimize as op
import get_data


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


def logistic_regression(X, y):

    m, n = X.shape

    initial_theta = np.zeros(n)

    log_reg = op.minimize(fun = cost_function, 
                          x0 = initial_theta, 
                          args = (X, y),
                          method = 'TNC',
                          jac = gradient)

    return(log_reg)


if __name__ == '__main__':

    adult = get_data.get_data()

    print(adult.head())

    adult_X, adult_y = get_data.data_to_np(adult)

    log_reg = logistic_regression(X = adult_X, y = adult_y)

    print(log_reg)
