import pandas as pd
import numpy as np
import scipy.optimize as op
from sklearn.preprocessing import StandardScaler



class LogisticRegression():
    """Base logistic regression class.

    Implements logisitic regression without any regularisation. Other implementations
    with regularisation inherit from this class.

    Parameters
    ----------
    fit_intercept : bool, default = True
        Should an intercept be included in the model?

    standardise : bool, default = True
        Should the inputs be standardised with sklearn's StandardScaler before fitting the model?

    """

    def __init__(self, fit_intercept = True, standardise = True):

        if not type(fit_intercept) is bool:
            
            raise TypeError('fit_intercept must be bool')

        if not type(standardise) is bool:
            
            raise TypeError('fit_intercept must be bool')

        self.fit_intercept = fit_intercept
        self.standardise = standardise


    def fit(self, X, y):
        """Function to fit model to given explanatory variables (X) and response variable (y).
        
        Function uses scipy.optimize.minimize to minimise self.cost_function.

        Parameters
        ----------
        X : np.ndarray
            2d array of explanatory variables to fit model on.

        y : np.ndarray
            1d array of response variable.

        """

        self.coefficient_names = [f'x{i}' for i in range(1, X.shape[1] + 1)]

        if self.standardise:

            self.scaler = StandardScaler()

            self.scaler.fit(X)

            X = self.scaler.transform(X)

        if self.fit_intercept:

            self.coefficient_names = ['intercept'] + self.coefficient_names

            X = np.hstack([np.ones((X.shape[0], 1)), X])

        self.m, self.n = X.shape

        initial_theta = np.zeros(self.n)

        # using optimiser suggested by stackoverflow user chammu;
        # https://stackoverflow.com/questions/18801002/fminunc-alternate-in-numpy
        self.optimisation_results = op.minimize(
            fun = self.cost_function, 
            x0 = initial_theta, 
            args = (X, y),
            method = 'TNC',
            jac = self.gradient
        )

        # extract coefficients into nice table
        if self.standardise:

            self.coefficients = pd.DataFrame(
                {
                    'name': self.coefficient_names,
                    'std_coef': self.optimisation_results['x']
                }
            )

            if self.fit_intercept:

                # calculate non standardised intercept term; 
                # beta_0 - sum((beta_i * m_i)/ s_i) for i in (1, n) where...
                # beta_0 is the standardised intercept
                # beta_i is the ith standardised coefficient
                # m_i is the mean for the ith variable, used for standardisation
                # s_i is the sd for the ith variable, used for standardisation
                non_std_intercept = self.optimisation_results['x'][0] - \
                    sum((np.array(self.optimisation_results['x'][1:]) * np.array(self.scaler.mean_)) / np.array(self.scaler.scale_))

                # divide standardised coefficients by scaling factors
                non_std_coefs = self.optimisation_results['x'][1:] / self.scaler.scale_

                non_std_coefs = [non_std_intercept] + non_std_coefs.tolist()

            else:

                non_std_coefs = self.optimisation_results['x'] / self.scaler.scale_

                non_std_coefs = non_std_coefs.tolist()

            self.coefficients['coef'] = non_std_coefs

        else:

            self.coefficients = pd.DataFrame(
                {
                    'name': self.coefficient_names,
                    'coef': self.optimisation_results['x']
                }
            )

        return self


    def cost_function(self, theta, X, y):
        '''Calculate the cost for given theta, X and y.
        
        Parameters
        ----------
        theta : np.ndarray
            Coefficient values.

        X : np.ndarray
            2d array of explanatory variables to fit model on.

        y : np.ndarray
            1d array of response variable.

        '''

        theta = theta.reshape((self.n, 1))
        
        y = y.reshape((self.m, 1))

        x_dot_theta = X.dot(theta)

        p = self.sigmoid(x_dot_theta)

        J = np.sum((-y * np.log(p)) - ((1 - y) * np.log(1 - p))) / self.m

        return J


    def gradient(self, theta, X, y):
        '''Calculate the gradient of the cost function w.r.t. each element of theta.
        
        Parameters
        ----------
        theta : np.ndarray
            Coefficient values.

        X : np.ndarray
            2d array of explanatory variables to fit model on.

        y : np.ndarray
            1d array of response variable.        
        
        '''
        
        theta = theta.reshape((self.n, 1))
        
        y = y.reshape((self.m, 1))

        grad = (X.T).dot(self.sigmoid(X.dot(theta)) - y) / self.m

        grad = grad.flatten()

        return grad


    def sigmoid(self, z):
        '''Calculate the sigmoid of all elements of a np array.
        
        Parameters
        ----------
        z : np.ndarray
            Array of values to transform.

        '''
        
        g = 1 / (1 + np.exp(-z))
        
        return g


    def predict_proba(self, X):
        """Function to return predictions from model for input data X.
        
        Parameters
        ----------
        X : np.ndarray
            2d array of explanatory variables to predict for.        

        """

        if self.fit_intercept:
            
            X = np.hstack([np.ones((X.shape[0], 1)), X])

        predictions = self.sigmoid(np.dot(X, self.coefficients['coef']))

        return predictions


