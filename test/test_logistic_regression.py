import unittest
import pandas as pd
import numpy as np
from numpy.testing import assert_almost_equal
import statsmodels.api as sm
from sklearn import preprocessing
from scipy import stats
from ADiscriminator import logistic_regression
from ADiscriminator import get_adult_dataset


class TestLogisticRegression(unittest.TestCase):

    def test_statsmodels_comparison(self):

        # fix to prevent error with depreceated scipy fcn being used by statsmodels
        # fix from github user VincentLa14;
        # https://github.com/statsmodels/statsmodels/issues/3931
        stats.chisqprob = lambda chisq, df: stats.chi2.sf(chisq, df)

        adult = get_adult_dataset.get_data()

        adult_X, adult_y = get_adult_dataset.data_to_np(adult)

        log_reg = logistic_regression.logistic_regression(X = adult_X, 
                                                          y = adult_y,
                                                          fit_intercept = True, 
                                                          standardise = True)

        adult_X = np.hstack([np.ones((adult_X.shape[0], 1)), adult_X])

        statsmodels_log_reg = sm.Logit(adult_y, adult_X)

        result = statsmodels_log_reg.fit()

        # using numpy testing to test almost equal for lists
        assert_almost_equal(actual = log_reg['coefficients']['coef'].tolist(),
                            desired = result.params.tolist(), 
                            decimal = 3)

    def test_2(self):
        self.assertTrue(True)
 
    # tests to add;
    # number of coefficients correct
    # intercept coefficient if specific
    # intercept coefficient in position 1
    # standardised coefficients if specified
    # with standardised coefs, non standardised can be transformed to standardised 



if __name__ == '__main__':
    unittest.main()

