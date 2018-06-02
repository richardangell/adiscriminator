import pandas as pd
import numpy as np
import scipy.optimize as op
from sklearn.preprocessing import StandardScaler
from ADiscriminator import get_adult_dataset
#import get_adult_dataset

def sigmoid(z):
    '''Calculate the sigmoid of all elements of a np array'''
    
    g = 1 / (1 + np.exp(-z))
    
    return(g)


def cost_function(theta, X, y):
    '''Calculate the cost for given theta, X and y'''
    
    m, n = X.shape
    
    theta = theta.reshape((n, 1))
    
    y = y.reshape((m,1))

    x_dot_theta = X.dot(theta)

    J = np.sum((-y * np.log(sigmoid(x_dot_theta))) - ((1 - y) * np.log(1 - sigmoid(x_dot_theta)))) / m
    
    return(J)


def gradient(theta, X, y):
    '''Calculate the gradient of the cost function w.r.t. each element of theta'''
    
    m, n = X.shape
    
    theta = theta.reshape((n, 1))
    
    y = y.reshape((m, 1))

    grad = (X.T).dot(sigmoid(X.dot(theta)) - y) / m
    
    return(grad.flatten())


def logistic_regression(X, y, fit_intercept = True, standardise = True):
    '''Fit a logistic regression model with optional standardisation and intercept'''

    model = {}

    model['fit_intercept'] = fit_intercept

    model['standardise'] = standardise

    coef_names = ['x' + str(i) for i in range(1, X.shape[1] + 1)]

    if standardise:

        scaler = StandardScaler()

        scaler.fit(X)

        X = scaler.transform(X)

        model['scaler'] = scaler

    if fit_intercept:

        coef_names = ['intercept'] + coef_names

        X = np.hstack([np.ones((X.shape[0], 1)), X])

    m, n = X.shape

    initial_theta = np.zeros(n)

    # using optimiser suggested by stackoverflow user chammu;
    # https://stackoverflow.com/questions/18801002/fminunc-alternate-in-numpy
    log_reg = op.minimize(fun = cost_function, 
                          x0 = initial_theta, 
                          args = (X, y),
                          method = 'TNC',
                          jac = gradient)

    model['optimisation_results'] = log_reg

    if standardise:

        model['coefficients'] = pd.DataFrame({'name': coef_names,
                                              'std_coef': log_reg['x']})

        if fit_intercept:

            # calculate non standardised intercept term; 
            # beta_0 - sum((beta_i * m_i)/ s_i) for i in (1, n) where...
            # beta_0 is the standardised intercept
            # beta_i is the ith standardised coefficient
            # m_i is the mean for the ith variable, used for standardisation
            # s_i is the sd for the ith variable, used for standardisation
            non_std_intercept = log_reg['x'][0] - \
                sum((np.array(log_reg['x'][1:]) * np.array(scaler.mean_)) / np.array(scaler.scale_))

            # divide standardised coefficients by scaling factors
            non_std_coefs = log_reg['x'][1:] / scaler.scale_

            non_std_coefs = [non_std_intercept] + non_std_coefs.tolist()

        else:

            non_std_coefs = log_reg['x'] / scaler.scale_

            non_std_coefs = non_std_coefs.tolist()

        model['coefficients']['coef'] = non_std_coefs

    else:

        model['coefficients'] = pd.DataFrame({'name': coef_names,
                                              'coef': log_reg['x']})

    return(model)


def predict_proba(model, X):

    return(np.dot(X, model['optimisation_result']['x']))




if __name__ == '__main__':

    adult = get_adult_dataset.get_data()

    adult_X, adult_y = get_adult_dataset.data_to_np(adult)

    log_reg = logistic_regression(X = adult_X, y = adult_y)

    print(log_reg['coefficients'])



