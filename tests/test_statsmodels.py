import pandas as pd
import numpy as np
import statsmodels.api as sm

from numpy.testing import assert_almost_equal

from adiscriminator import data



def test_compare_glm_logistic_regression():
    """Check that statsmodel Logit and binomial glm give the same results"""

    adult = data.get_data()

    adult_X, adult_y = data.data_to_np(adult)

    adult_X = np.hstack([np.ones((adult_X.shape[0], 1)), adult_X])

    log_reg = sm.Logit(adult_y, adult_X)

    log_reg_result = log_reg.fit()

    glm = sm.GLM(adult_y, adult_X, family = sm.families.Binomial(link = sm.families.links.logit))

    glm_result = glm.fit()

    assert_almost_equal(actual = log_reg_result.params.tolist(),
                        desired = glm_result.params.tolist(), 
                        decimal = 4)




