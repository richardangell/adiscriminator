from adiscriminator.logistic_regression.base import LogisticRegression



class RidgeRegression(LogisticRegression):
    """Ridge logistic regression class.

    Implements logisitic regression with L2 regularisation. 

    Parameters
    ----------
    fit_intercept : bool, default = True
        Should an intercept be included in the model?

    standardise : bool, default = True
        Should the inputs be standardised with sklearn's StandardScaler before fitting the model?

    lambda_ : int or float, default = 0
        Regularisation strength. Larger values penalise coefficient values more.

    penalised_intercept : bool default = False
        Should the intercept term be penalised as well as coefficients for explanatory variables?

    """

    def __init__(self, fit_intercept = True, standardise = True, lambda_ = 0, penalise_intercept = False):

        super().__init__(fit_intercept = fit_intercept, standardise = standardise)

        if not type(penalise_intercept) is bool:
            
            raise TypeError('penalise_intercept must be bool')

        self.penalise_intercept = penalise_intercept

        self.lambda_ = lambda_


    def cost_function(self, theta, X, y):
        '''Calculate the cost with l2 penalty for given theta, X and y.
        
        Parameters
        ----------
        theta : np.ndarray
            Coefficient values.

        X : np.ndarray
            2d array of explanatory variables to fit model on.

        y : np.ndarray
            1d array of response variable.

        '''
        
        J = super().cost_function(theta, X, y)
        
        if self.penalise_intercept:

            penalty_term = self.lambda_ * sum(theta ** 2) / (2 * self.m)

        else:

            penalty_term = self.lambda_ * sum(theta[1:] ** 2) / (2 * self.m)
        
        J = J + penalty_term

        return J


    def gradient(self, theta, X, y):
        '''Calculate the gradient of the cost function including l2 penalty term w.r.t. each element of theta.
        
        Parameters
        ----------
        theta : np.ndarray
            Coefficient values.

        X : np.ndarray
            2d array of explanatory variables to fit model on.

        y : np.ndarray
            1d array of response variable.   

        '''

        grad = J = super().gradient(theta, X, y)

        penalty_term = (self.lambda_ / self.m) * theta

        if not self.penalise_intercept:

            penalty_term[0] = 0

        grad = grad + penalty_term

        grad = grad.flatten()

        return grad


