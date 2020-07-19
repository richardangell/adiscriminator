import numpy as np
import scipy
from sklearn.preprocessing import StandardScaler

import pytest
from pytest_mock import mocker
from numpy.testing import assert_array_equal

import adiscriminator as ad



class TestFit(object):
    """Tests for the fit method."""

    def setup_class(self):
        """Load data to build models on."""

        adult = ad.data.get_data()
        self.X, self.y = ad.data.data_to_np(adult)


    def test_data_standardised_no_intercept(self, mocker):
        """Test that X is standardised if standardise is True, without an intercept fit."""

        model = ad.logistic_regression.base.LogisticRegression(standardise = True, fit_intercept = False)

        scaled_X = StandardScaler().fit_transform(self.X)

        spy = mocker.spy(scipy.optimize, 'minimize')

        model.fit(self.X, self.y)

        call_kwargs = spy.call_args_list[0][1]

        # keyword arg 'arg' in call to scipy.optimize.minimize
        call_arg_kwarg = call_kwargs['args']

        # X is first item in 'arg' tuple
        call_X = call_arg_kwarg[0]

        assert_array_equal(call_X, scaled_X)


    def test_data_standardised_with_intercept(self, mocker):
        """Test that X is standardised if standardise is True, with an intercept fit."""

        model = ad.logistic_regression.base.LogisticRegression(standardise = True, fit_intercept = True)

        scaled_X = StandardScaler().fit_transform(self.X)

        spy = mocker.spy(scipy.optimize, 'minimize')

        model.fit(self.X, self.y)

        call_kwargs = spy.call_args_list[0][1]

        # keyword arg 'arg' in call to scipy.optimize.minimize
        call_arg_kwarg = call_kwargs['args']

        # X is first item in 'arg' tuple
        call_X = call_arg_kwarg[0]

        # constant term is added in first column if fit_intercept is True
        call_X_without_intercept = call_X[:,1:]

        assert_array_equal(call_X_without_intercept, scaled_X)


    def test_X_no_intercept(self, mocker):
        """Test X used in fitting is expected if fit_intercept is False."""

        model = ad.logistic_regression.base.LogisticRegression(standardise = False, fit_intercept = False)

        spy = mocker.spy(scipy.optimize, 'minimize')

        model.fit(self.X, self.y)

        call_kwargs = spy.call_args_list[0][1]

        call_arg_kwarg = call_kwargs['args']

        call_X = call_arg_kwarg[0]

        assert_array_equal(call_X, self.X)


    def test_X_with_intercept(self, mocker):
        """Test X used in fitting is expected if fit_intercept is True."""

        model = ad.logistic_regression.base.LogisticRegression(standardise = False, fit_intercept = True)

        spy = mocker.spy(scipy.optimize, 'minimize')

        model.fit(self.X, self.y)

        call_kwargs = spy.call_args_list[0][1]

        call_arg_kwarg = call_kwargs['args']

        call_X = call_arg_kwarg[0]

        call_X_without_intercept = call_X[:,1:]

        # check the non intercept columns are the same
        assert_array_equal(call_X_without_intercept, self.X)
                
        # check intercept is all ones of correct shape
        assert_array_equal(call_X[:, 0], np.ones((self.X.shape[0],)))


    def test_scipy_minimise_call(self, mocker):
        """Test scipy.optimize.minimize is called with correct args (excluding X) and correct # times."""

        spy = mocker.spy(scipy.optimize, 'minimize')

        model = ad.logistic_regression.base.LogisticRegression(standardise = False, fit_intercept = False)

        model.fit(self.X, self.y)

        assert spy.call_count == 1, \
            f'Unexpected number of calls to scipy.optimize.minimize - expecting 1 but got {spy.call_count}'

        call_pos_args = spy.call_args_list[0][0]

        assert call_pos_args == (), 'Positional arguments not expected in scipy.optimize.minimize call'

        call_kwargs = spy.call_args_list[0][1]

        assert call_kwargs['fun'] == model.cost_function, \
            """Unexpected 'fun' kwarg in scipy.optimize.minimize call"""

        # no intercept fit so initial theta is # columns 
        assert_array_equal(call_kwargs['x0'], np.zeros(self.X.shape[1])), \
            """Unexpected 'x0' kwarg in scipy.optimize.minimize call"""

        # note, the X element has been tested in the 4 tests above
        assert_array_equal(call_kwargs['args'][1], self.y), \
            """Unexpected y element of 'args' kwarg in scipy.optimize.minimize call"""

        assert call_kwargs['method'] == 'TNC', \
            """Unexpected 'method' kwarg in scipy.optimize.minimize call"""

        assert call_kwargs['jac'] == model.gradient, \
            """Unexpected 'jac' kwarg in scipy.optimize.minimize call"""

    @pytest.mark.parametrize(
        "standardise,expected_cols", 
        [
            (True, ['name', 'std_coef', 'coef']), 
            (False, ['name', 'coef'])
        ]
    )
    def test_coefficient_table_columns(self, standardise, expected_cols):
        """Test that the model coefficients table has the correct columns."""

        model = ad.logistic_regression.base.LogisticRegression(standardise = standardise, fit_intercept = True)

        model.fit(self.X, self.y)

        assert list(model.coefficients.columns) == expected_cols, \
            f'Unexpected columns in model coefficients table with standardise = {standardise}'


    @pytest.mark.parametrize(
        "fit_intercept,expected_additional_coefficients", 
        [
            (True, 1), 
            (False, 0)
        ]
    )
    def test_number_coefficients(self, fit_intercept, expected_additional_coefficients):
        """Test that the model coefficients table has the correct columns."""

        model = ad.logistic_regression.base.LogisticRegression(standardise = False, fit_intercept = fit_intercept)

        model.fit(self.X, self.y)

        assert model.coefficients.shape[0] == self.X.shape[1] + expected_additional_coefficients, \
            f'Unexpected number of coefficients in model coefficients table with fit_intercept = {fit_intercept}'

    @pytest.mark.parametrize(
        "standardise", 
        [
            True,
            False
        ]
    )
    def test_intercept_coefficient_in_table(self, standardise):
        """Test the intercept coefficient appears in the first row of the intercept table, fit_intercept is True."""

        model = ad.logistic_regression.base.LogisticRegression(standardise = standardise, fit_intercept = True)

        model.fit(self.X, self.y)

        assert model.coefficients['name'][0] == 'intercept', \
            f'Intercept term not in coefficients table when standardise = {standardise}'


    def test_standardised_coefficient_values_with_intercept(self):
        """Test standardised coefficient values are calculated correctly."""

        model = ad.logistic_regression.base.LogisticRegression(standardise = True, fit_intercept = True)

        model.fit(self.X, self.y)

        X_scaler = StandardScaler().fit(self.X)

        # yhati = b0 + sum j [bj * ((xij - xbarj) / sj)]
        #       = (b0 - sum j [xbarj * bj / sj])  + sum j [(bj / sj) * xij] 
        #       = b'0 + sum j [b'j * xij]
        # hence..
        # for non intercept values; b'j = bj / sj
        # and intercept value; b'0 = b0 - sum j [xbarj * bj / sj]
        # where 
        # b'j is the jth non standardised coefficient
        # bj is the jth standardised coefficient
        # sj is the std for the jth variable
        # xbarj is the mean for jth variable
        # xij is the value of the jth variable for the ith record

        assert_array_equal(
            model.coefficients['std_coef'][1:] / X_scaler.scale_, 
            model.coefficients['coef'][1:]
        ) 

        beta0_offset = sum((np.array(model.coefficients['std_coef'][1:]) * np.array(X_scaler.mean_)) / np.array(X_scaler.scale_))

        assert model.coefficients['std_coef'][0] - beta0_offset == model.coefficients['coef'][0], \
            f'Incorrect standardised beta0 value'


    def test_standardised_coefficient_values(self):
        """Test standardised coefficient values are calculated correctly."""

        model = ad.logistic_regression.base.LogisticRegression(standardise = True, fit_intercept = False)

        model.fit(self.X, self.y)

        assert_array_equal(
            model.coefficients['std_coef'] / StandardScaler().fit(self.X).scale_, 
            model.coefficients['coef']
        ) 
