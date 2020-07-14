import numpy as np

from adiscriminator.logistic_regression.base import LogisticRegression



class GroupMeanEqualisingRegression(LogisticRegression):
    """Ridge logistic regression class.

    Implements logisitic regression with L2 regularisation. 

    Parameters
    ----------
    group : np.ndarray
        Group membership variable, can only have 2 levels 0 and 1 currently.

    fit_intercept : bool, default = True
        Should an intercept be included in the model?

    standardise : bool, default = True
        Should the inputs be standardised with sklearn's StandardScaler before fitting the model?

    lambda_ : int or float, default = 0
        Regularisation strength. Larger values penalise coefficient values more.
        
    """

    def __init__(self, group, fit_intercept = True, standardise = True, lambda_ = 0):

        super().__init__(fit_intercept = fit_intercept, standardise = standardise)

        self.lambda_ = lambda_

        if len(np.unique(group)) > 2:

            raise NotImplementedError('group can have a max of 2 levels')

        self.group = group


    def fit(self, X, y):
        """Function to fit model to given explanatory variables (X) and response variable (y).
        
        Function calls base.LogisticRegression.fit() then deletes the group attribute.

        Parameters
        ----------
        X : np.ndarray
            2d array of explanatory variables to fit model on.

        y : np.ndarray
            1d array of response variable.

        """

        if not len(self.group) == X.shape[0]:
            
            raise ValueError('group must be the same length as X')

        base_fit = super().fit(X, y)

        delattr(self, 'group')

        return base_fit


    def cost_function(self, theta, X, y):
        '''Calculate the cost for given theta, X and y.

        Function includes penalty for differences between mean of groups.
        
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

        p = super().calculate_p(theta, X)

        d = self.calculate_group_mean_differences(p, self.group)

        group_mean_difference_penalty = -np.log(1 - d ** 2)

        J = J + self.lambda_ * group_mean_difference_penalty

        return J


    def calculate_group_mean_differences(self, p, g):
        """Calculate the difference in average prediction by groups.
        
        Parameters
        ----------
        p : np.ndarray
            Predictions.

        g : np.ndarray
            Group membership variable.
        
        """

        g1_loc = g == 0

        g1_weight = np.sum(g1_loc)

        g2_weight = len(g) - g1_weight  

        g1_ave = np.sum(p[g1_loc]) / g1_weight

        g2_ave = np.sum(p[np.invert(g1_loc)]) / g2_weight

        d = g1_ave - g2_ave

        return d


    def gradient(self, theta, X, y):
        '''Calculate the gradient of the cost function w.r.t. each element of theta. 

        Also includes gradient accounting for the difference in group means penalty term.
        
        Parameters
        ----------
        theta : np.ndarray
            Coefficient values.

        X : np.ndarray
            2d array of explanatory variables to fit model on.

        y : np.ndarray
            1d array of response variable.        
        
        '''

        grad = super().gradient(theta, X, y)

        grad_penalty = self.gradient_group_mean_differences(theta, X, y)

        grad = grad + grad_penalty

        return grad


    def gradient_group_mean_differences(self, theta, X, y):
        '''Calculate the gradient for the difference in group means penalty term only. 

        Derivation can be found in the derivation.md file.
        
        Parameters
        ----------
        theta : np.ndarray
            Coefficient values.

        X : np.ndarray
            2d array of explanatory variables to fit model on.

        y : np.ndarray
            1d array of response variable.        
        
        '''

        p = super().calculate_p(theta, X)

        d = self.calculate_group_mean_differences(p, self.group)

        p_one_minus_p = p * (1 - p)

        p_one_minus_p_X = p_one_minus_p * X

        g1_loc = self.group == 0

        g1_weight = np.sum(g1_loc)

        g2_weight = len(self.group) - g1_weight  

        g1_gradient = p_one_minus_p_X[g1_loc].sum(axis = 0) / g1_weight 

        g2_gradient = p_one_minus_p_X[np.invert(g1_loc)].sum(axis = 0) / g2_weight

        grad = -2 * d * self.lambda_ * (g1_gradient - g2_gradient)  / (1 - d ** 2) 

        return grad

