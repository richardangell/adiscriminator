import numpy as np
import scipy
from sklearn.preprocessing import StandardScaler

import pytest
from pytest_mock import mocker
from numpy.testing import assert_array_equal

import adiscriminator as ad
from adiscriminator.logistic_regression.base import LogisticRegression
from adiscriminator.logistic_regression.ridge import RidgeRegression
from adiscriminator.logistic_regression.fair import GroupMeanEqualisingRegression



def initialise_model(cls, standardise, fit_intercept):
    """Function to initialise classes from logisitic_regression module."""

    if cls is GroupMeanEqualisingRegression:

        # create random group as we're not interested in learnt model values in this script
        np.random.seed(1)
        random_group = (np.random.rand(32560) > 0.5).astype(int)

        obj = cls(group = random_group, standardise = standardise, fit_intercept = fit_intercept)

    else:

        obj = cls(standardise = standardise, fit_intercept = fit_intercept)

    return obj


@pytest.mark.parametrize(
    "cls", 
    [
        LogisticRegression, 
        RidgeRegression, 
        GroupMeanEqualisingRegression,   
    ]
)
class TestFit():
    """Tests for the fit method on model classes."""

    def setup_class(self):
        """Load data to build models on."""

        adult = ad.data.get_data()
        self.X, self.y = ad.data.data_to_np(adult)


    def test_data_standardised_no_intercept(self, cls, mocker):
        """Test that X is standardised if standardise is True, without an intercept fit."""

        model = initialise_model(cls = cls, standardise = True, fit_intercept = False)

        scaled_X = StandardScaler().fit_transform(self.X)

        spy = mocker.spy(scipy.optimize, 'minimize')

        model.fit(self.X, self.y)

        call_kwargs = spy.call_args_list[0][1]

        # keyword arg 'arg' in call to scipy.optimize.minimize
        call_arg_kwarg = call_kwargs['args']

        # X is first item in 'arg' tuple
        call_X = call_arg_kwarg[0]

        assert_array_equal(call_X, scaled_X)


    def test_data_standardised_with_intercept(self, cls, mocker):
        """Test that X is standardised if standardise is True, with an intercept fit."""

        model = initialise_model(cls = cls, standardise = True, fit_intercept = True)

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


    def test_X_no_intercept(self, cls, mocker):
        """Test X used in fitting is expected if fit_intercept is False."""

        model = initialise_model(cls = cls, standardise = False, fit_intercept = False)

        spy = mocker.spy(scipy.optimize, 'minimize')

        model.fit(self.X, self.y)

        call_kwargs = spy.call_args_list[0][1]

        call_arg_kwarg = call_kwargs['args']

        call_X = call_arg_kwarg[0]

        assert_array_equal(call_X, self.X)


    def test_X_with_intercept(self, cls, mocker):
        """Test X used in fitting is expected if fit_intercept is True."""

        model = initialise_model(cls = cls, standardise = False, fit_intercept = True)

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


    def test_scipy_minimise_call(self, cls, mocker):
        """Test scipy.optimize.minimize is called with correct args (excluding X) and correct # times."""

        spy = mocker.spy(scipy.optimize, 'minimize')

        model = initialise_model(cls = cls, standardise = False, fit_intercept = False)

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
    def test_coefficient_table_columns(self, cls, standardise, expected_cols):
        """Test that the model coefficients table has the correct columns."""

        model = initialise_model(cls = cls, standardise = standardise, fit_intercept = True)

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
    def test_number_coefficients(self, cls, fit_intercept, expected_additional_coefficients):
        """Test that the model coefficients table has the correct columns."""

        model = initialise_model(cls = cls, standardise = False, fit_intercept = fit_intercept)

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
    def test_intercept_coefficient_in_table(self, cls, standardise):
        """Test the intercept coefficient appears in the first row of the intercept table, fit_intercept is True."""

        model = initialise_model(cls = cls, standardise = standardise, fit_intercept = True)

        model.fit(self.X, self.y)

        assert model.coefficients['name'][0] == 'intercept', \
            f'Intercept term not in coefficients table when standardise = {standardise}'


    def test_standardised_coefficient_values_with_intercept(self, cls):
        """Test standardised coefficient values are calculated correctly."""

        model = initialise_model(cls = cls, standardise = True, fit_intercept = True)

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


    def test_standardised_coefficient_values(self, cls):
        """Test standardised coefficient values are calculated correctly."""

        model = initialise_model(cls = cls, standardise = True, fit_intercept = False)

        model.fit(self.X, self.y)

        assert_array_equal(
            model.coefficients['std_coef'] / StandardScaler().fit(self.X).scale_, 
            model.coefficients['coef']
        ) 



@pytest.mark.parametrize(
    "cls", 
    [
        LogisticRegression, 
        RidgeRegression, 
        GroupMeanEqualisingRegression,   
    ]
)
class TestCostFunction():
    """Tests for the cost_function method on model classes."""

    def setup_class(self):
        """Load data to build models on."""

        adult = ad.data.get_data()
        self.X, self.y = ad.data.data_to_np(adult)


    def test_return_value(self, cls):
        """Test return value is the correct type (and shape)."""

        model = initialise_model(cls = cls, standardise = True, fit_intercept = True)

        # set m as it is usually set in fit
        model.m, model.n = self.X.shape

        dummy_theta = np.zeros(self.X.shape[1])

        J = model.cost_function(
            theta = dummy_theta,
            X = self.X, 
            y = self.y
        )

        if isinstance(J, np.ndarray):

            pytest.fail('J should be not be a np.ndarray')           

        assert isinstance(J , np.float), \
            f'cost_function does not return expected type (np.float) got {type(J)}'



@pytest.mark.parametrize(
    "cls", 
    [
        LogisticRegression, 
        RidgeRegression, 
        GroupMeanEqualisingRegression,   
    ]
)
class TestCalculateP():
    """Tests for the calculate_p method on model classes."""

    def setup_class(self):
        """Load data to build models on."""

        adult = ad.data.get_data()
        self.X, self.y = ad.data.data_to_np(adult)


    def test_return_shape(self, cls):
        """Test return value is the correct shape."""

        model = initialise_model(cls = cls, standardise = True, fit_intercept = True)

        # set m as it is usually set in fit
        model.m, model.n = self.X.shape

        dummy_theta = np.zeros(self.X.shape[1])

        p = model.calculate_p(
            theta = dummy_theta,
            X = self.X
        )

        assert p.shape == (self.X.shape[0], 1), 'Incorrect shape for calculate_p output'



@pytest.mark.parametrize(
    "cls", 
    [
        LogisticRegression, 
        RidgeRegression, 
        GroupMeanEqualisingRegression,   
    ]
)
class TestGradient():
    """Tests for the gradient method on model classes."""

    def setup_class(self):
        """Load data to build models on."""

        adult = ad.data.get_data()
        self.X, self.y = ad.data.data_to_np(adult)


    def test_return_shape(self, cls):
        """Test return value is the correct shape."""

        model = initialise_model(cls = cls, standardise = True, fit_intercept = True)

        # set m as it is usually set in fit
        model.m, model.n = self.X.shape

        dummy_theta = np.zeros(self.X.shape[1])

        grad = model.gradient(
            theta = dummy_theta,
            X = self.X, 
            y = self.y
        )

        assert grad.shape == (self.X.shape[1], ), 'Incorrect shape for gradient output'



@pytest.mark.parametrize(
    "cls", 
    [
        LogisticRegression, 
        RidgeRegression, 
        GroupMeanEqualisingRegression,   
    ]
)
class TestSigmoid():
    """Tests for the sigmoid method on model classes."""

    def setup_class(self):
        """Load data to build models on."""

        adult = ad.data.get_data()
        self.X, self.y = ad.data.data_to_np(adult)


    def test_return_shape(self, cls):
        """Test return value is the correct shape."""

        model = initialise_model(cls = cls, standardise = True, fit_intercept = True)

        sigmoid_X = model.sigmoid(
            z = self.X
        )

        assert sigmoid_X.shape == self.X.shape, 'Incorrect shape for sigmoid output'



@pytest.mark.parametrize(
    "cls", 
    [
        LogisticRegression, 
        RidgeRegression, 
        GroupMeanEqualisingRegression,   
    ]
)
class TestPredictProba():
    """Tests for the predict_proba method on model classes."""

    def setup_class(self):
        """Load data to build models on."""

        adult = ad.data.get_data()
        self.X, self.y = ad.data.data_to_np(adult)


    def test_return_shape(self, cls):
        """Test return value is the correct shape."""

        model = initialise_model(cls = cls, standardise = True, fit_intercept = True)

        model.fit(self.X, self.y)

        predictions = model.predict_proba(self.X)

        assert predictions.shape == (self.X.shape[0], 1), 'Incorrect shape for predict_proba output'







