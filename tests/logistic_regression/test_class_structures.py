import pytest
import inspect

from adiscriminator.logistic_regression.base import LogisticRegression
from adiscriminator.logistic_regression.ridge import RidgeRegression
from adiscriminator.logistic_regression.fair import GroupMeanEqualisingRegression


def initialise_class(cls):
    """Function to initialise classes from logisitic_regression module."""

    if cls is GroupMeanEqualisingRegression:

        obj = cls(group = 1)

    else:

        obj = cls()

    return obj


@pytest.mark.parametrize(
    "cls", 
    [
        LogisticRegression, 
        RidgeRegression, 
        GroupMeanEqualisingRegression,   
    ]
)
def test_class_inheritance(cls):

    obj = initialise_class(cls)

    assert isinstance(obj, LogisticRegression)



@pytest.mark.parametrize(
    "cls", 
    [
        LogisticRegression,
        RidgeRegression,
        GroupMeanEqualisingRegression    
    ]
)
@pytest.mark.parametrize(
    "name", 
    [
        'fit',
        'cost_function',
        'gradient'        
    ]
)
def test_class_methods(cls, name):
    """Test that method exists on class instance."""

    obj = initialise_class(cls)

    assert hasattr(obj, name), f'{obj} does not have method {name}' 
    assert inspect.ismethod(getattr(obj, name)), f'{obj} attribute {name} is not a method' 


@pytest.mark.parametrize(
    "cls", 
    [
        LogisticRegression, 
        RidgeRegression, 
        GroupMeanEqualisingRegression,   
    ]
)
@pytest.mark.parametrize(
    "method_name,expected_args", 
    [
        ('fit', ['self', 'X', 'y']), 
        ('cost_function', ['self', 'theta', 'X', 'y']), 
        ('calculate_p', ['self', 'theta', 'X']), 
        ('gradient', ['self', 'theta', 'X', 'y']),
        ('sigmoid', ['self', 'z']),
        ('predict_proba', ['self', 'X'])
    ]
)
def test_fit_args(cls, method_name, expected_args):

    obj = initialise_class(cls)
    obj_method = getattr(obj, method_name)

    method_args = inspect.getfullargspec(obj_method).args

    assert method_args == expected_args
