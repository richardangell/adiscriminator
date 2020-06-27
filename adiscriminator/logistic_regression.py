import pandas as pd
import numpy as np
import scipy.optimize as op
from sklearn.preprocessing import StandardScaler
from adiscriminator import get_adult_dataset
from sklearn.linear_model import LogisticRegression
import statsmodels.api as sm
from sklearn import preprocessing


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


def cost_function_l2(theta, X, y, lambda_, include_first_coef):
    '''Calculate the cost with l2 penalty for given theta, X and y'''
    
    m, n = X.shape
    
    theta = theta.reshape((n, 1))
    
    y = y.reshape((m,1))

    x_dot_theta = X.dot(theta)

    J = np.sum((-y * np.log(sigmoid(x_dot_theta))) - ((1 - y) * np.log(1 - sigmoid(x_dot_theta)))) / m
    
    if include_first_coef:

        penalty_term = lambda_ * sum(theta ** 2) / (2 * m)

    else:

        penalty_term = lambda_ * sum(theta[1:] ** 2) / (2 * m)
    
    J = J + penalty_term

    return(J)


def cost_function_adiscriminator_cat(theta, X, y, adiscriminator):

    J = cost_function(theta, X, y)
    
    discrimination_penalty = cost_differential(theta, X, y, adiscriminator)

    print('J:', J, 'pen:', discrimination_penalty, 'total:',  J + discrimination_penalty)

    J = J + discrimination_penalty
  
    return(J)


def cost_differential(theta, X, y, adiscriminator):

    m, n = X.shape
    
    theta = theta.reshape((n, 1))
    
    y = y.reshape((m,1))

    x_dot_theta = X.dot(theta)

    idx_g1 = adiscriminator == 0

    weight_g1 = np.sum(idx_g1)

    weight_g2 = len(adiscriminator) - weight_g1  

    preds = sigmoid(x_dot_theta)

    g1_ave = np.sum(preds[idx_g1]) / weight_g1

    g2_ave = np.sum(preds[np.invert(idx_g1)]) / weight_g2

    sq_diff = (g1_ave - g2_ave) ** 2

    print(g1_ave, g2_ave,sq_diff)

    return(30 * -np.log(1 - sq_diff))



def gradient(theta, X, y):
    '''Calculate the gradient of the cost function w.r.t. each element of theta'''
    
    m, n = X.shape
    
    theta = theta.reshape((n, 1))
    
    y = y.reshape((m, 1))

    grad = (X.T).dot(sigmoid(X.dot(theta)) - y) / m

    return(grad.flatten())


def gradient_l2(theta, X, y, lambda_, include_first_coef):
    '''Calculate the gradient of the cost function including l2 penalty term w.r.t. each element of theta'''

    m, n = X.shape
    
    theta = theta.reshape((n, 1))
    
    y = y.reshape((m, 1))

    grad = (X.T).dot(sigmoid(X.dot(theta)) - y) / m

    penalty_term = (lambda_ / m) * theta

    if not include_first_coef:

        penalty_term[0] = 0

    grad = grad + penalty_term

    return(grad.flatten())


def gradient_adiscriminator_cat(theta, X, y, adiscriminator):

    grad = gradient(theta, X, y)

    grad_penalty = gradient_differential(theta, X, y, adiscriminator)

    grad = grad + grad_penalty

    return(grad)





def gradient_differential(theta, X, y, adiscriminator):

    m, n = X.shape
    
    theta = theta.reshape((n, 1))
    
    y = y.reshape((m, 1))

    preds = sigmoid(X.dot(theta))

    idx_g1 = adiscriminator == 0

    weight_g1 = np.sum(idx_g1)

    weight_g2 = len(adiscriminator) - weight_g1  

    preds2 = preds * (1 - preds)

    preds3 = preds2 * X

    group_mult = idx_g1.astype(int)

    group_mult[group_mult == 0] = -1

    group_mult = group_mult.reshape((m ,1))

    preds4 = group_mult * preds3 

    pt1 = (preds4[idx_g1].sum(axis = 0) / weight_g1)

    pt2 = (preds4[np.invert(idx_g1)].sum(axis = 0) / weight_g2)

    return(pt1 + pt2)


def logistic_regression(X, y, 
                        fit_intercept = True, 
                        standardise = True, 
                        regularisation = None, 
                        lambda_ = 0, 
                        penalise_intercept = None,
                        adiscriminator_column = None):
    '''Fit a logistic regression model with optional standardisation and intercept'''

    assert isinstance(fit_intercept, bool), 'fit_intercept must be bool'

    assert isinstance(standardise, bool), 'standardise must be bool'

    regularisation_valid = ['l1', 'l2', None]

    assert regularisation in regularisation_valid, \
        'regularisation must be one of %s' % (regularisation_valid)

    if regularisation == 'l1':

        raise ValueError('l1 regularisation not yet supported.')

    penalise_intercept_valid = [True, False, None]

    assert penalise_intercept in penalise_intercept_valid, \
        'penalise_intercept must be one of %s' % (penalise_intercept_valid)

    if adiscriminator_column is not None:

        assert len(adiscriminator_column) == X.shape[0], 'adiscriminator_column must be the same length as X'

    # by default if an intercept is included in the model do not include it in the penalisation
    if penalise_intercept is None:

        penalise_intercept = not(fit_intercept)

    model = {}

    model['fit_intercept'] = fit_intercept

    model['standardise'] = standardise

    model['regularisation'] = regularisation

    model['lambda_'] = lambda_

    model['penalise_intercept'] = penalise_intercept

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

    if (regularisation is None) & (adiscriminator_column is None):

        # using optimiser suggested by stackoverflow user chammu;
        # https://stackoverflow.com/questions/18801002/fminunc-alternate-in-numpy
        log_reg = op.minimize(fun = cost_function, 
                              x0 = initial_theta, 
                              args = (X, y),
                              method = 'TNC',
                              jac = gradient)

    elif regularisation is not None:

        log_reg = op.minimize(fun = cost_function_l2, 
                              x0 = initial_theta, 
                              args = (X, y, lambda_, penalise_intercept),
                              method = 'TNC',
                              jac = gradient_l2)

    elif adiscriminator_column is not None:

        log_reg = op.minimize(fun = cost_function_adiscriminator_cat, 
                              x0 = initial_theta, 
                              args = (X, y, adiscriminator_column),
                              method = 'TNC',
                              jac = gradient_adiscriminator_cat)

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

    return(sigmoid(np.dot(X, model['coefficients']['coef'])))


if __name__ == '__main__':

    print('-----')

    adult = get_adult_dataset.get_data()

    adult_X, adult_y = get_adult_dataset.data_to_np(adult)

    #log_reg = logistic_regression(X = adult_X, y = adult_y)

    log_reg = logistic_regression(X = adult_X, y = adult_y, regularisation = 'l2', lambda_ = 5, penalise_intercept = True)

    print(log_reg['coefficients'])

    adult_X2 = preprocessing.scale(adult_X)

    sk_log_reg2 = LogisticRegression(C = 1/5, penalty = 'l2', fit_intercept = True)

    sk_log_reg2.fit(adult_X2, adult_y)

    print(sk_log_reg2.intercept_)

    print(sk_log_reg2.coef_[0])

    gender_col = np.array((adult.sex == ' Female').astype(int))

    dis_log_reg = logistic_regression(X = adult_X, y = adult_y, adiscriminator_column = gender_col)

    print(dis_log_reg['coefficients'])
