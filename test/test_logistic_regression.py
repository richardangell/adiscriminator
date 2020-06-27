import unittest
import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
from numpy.testing import assert_almost_equal
from adiscriminator import logistic_regression
from adiscriminator import get_adult_dataset


class TestLogisticRegression(unittest.TestCase):

    # tests to add;
    # number of coefficients correct
    # intercept coefficient if specified
    # intercept coefficient in position 1
    # standardised coefficients if specified
    # with standardised coefs, non standardised can be transformed to standardised 

    def test_1(self):

        self.assertTrue(True)
 
    def test_compare_statsmodels_non_reg(self):
        """Compare statsmodels logistic regression to non regularised logistic_regression"""

        # fix to prevent error with depreceated scipy fcn being used by statsmodels
        # fix from github user VincentLa14;
        # https://github.com/statsmodels/statsmodels/issues/3931
        stats.chisqprob = lambda chisq, df: stats.chi2.sf(chisq, df)

        adult = get_adult_dataset.get_data()

        adult_X, adult_y = get_adult_dataset.data_to_np(adult)

        log_reg = logistic_regression.logistic_regression(X = adult_X, 
                                                          y = adult_y,
                                                          fit_intercept = True, 
                                                          standardise = True,
                                                          regularisation = None,
                                                          lambda_ = 0, 
                                                          penalise_intercept = None)

        # statsmodels requires a constant column to be added to fit an intercept
        adult_X = np.hstack([np.ones((adult_X.shape[0], 1)), adult_X])

        statsmodels_log_reg = sm.Logit(adult_y, adult_X)

        result = statsmodels_log_reg.fit()

        assert_almost_equal(actual = log_reg['coefficients']['coef'].tolist(),
                            desired = result.params.tolist(), 
                            decimal = 3)

    def test_compare_sklearn_l2_reg(self):
        """Compare scikit learn logistic regression to logistic_regression with l2 regularisation and intercept"""

        adult = get_adult_dataset.get_data()

        adult_X, adult_y = get_adult_dataset.data_to_np(adult)

        regularisation_value = 8

        # note scikit learn penalises the intercept term as well as all the other coefficients
        log_reg = logistic_regression.logistic_regression(X = adult_X, 
                                                          y = adult_y,
                                                          fit_intercept = True, 
                                                          standardise = True,
                                                          regularisation = 'l2',
                                                          lambda_ = regularisation_value, 
                                                          penalise_intercept = True)

        # non-standardised scikit learn coefficients don't seem to match
        # so check the standardised coefficients for now
        adult_X = preprocessing.scale(adult_X)

        # scikit learn's regularisation strength is C = 1 / lambda
        sklearn_log_reg = LogisticRegression(C = 1 / regularisation_value, penalty = 'l2', fit_intercept = True)

        sklearn_log_reg.fit(adult_X, adult_y)

        assert_almost_equal(actual = log_reg['coefficients']['std_coef'].tolist(),
                            desired = sklearn_log_reg.intercept_.tolist() + sklearn_log_reg.coef_[0].tolist(), 
                            decimal = 4)

    def test_compare_sklearn_l2_reg_no_intercept(self):
        """Compare scikit learn logistic regression to logistic_regression with l2 regularisation and no intercept term"""

        adult = get_adult_dataset.get_data()

        adult_X, adult_y = get_adult_dataset.data_to_np(adult)

        regularisation_value = 4

        log_reg = logistic_regression.logistic_regression(X = adult_X, 
                                                          y = adult_y,
                                                          fit_intercept = False, 
                                                          standardise = True,
                                                          regularisation = 'l2',
                                                          lambda_ = regularisation_value, 
                                                          penalise_intercept = None)

        adult_X = preprocessing.scale(adult_X)

        sklearn_log_reg = LogisticRegression(C = 1 / regularisation_value, penalty = 'l2', fit_intercept = False)

        sklearn_log_reg.fit(adult_X, adult_y)

        assert_almost_equal(actual = log_reg['coefficients']['std_coef'].tolist(),
                            desired = sklearn_log_reg.coef_[0].tolist(), 
                            decimal = 4)


if __name__ == '__main__':

    unittest.main()

