from nose.tools import *
import numpy as np

from ..causal import CausalModel
from ..core.propensity import PropensitySelect
from tools import random_data_base


def test_get_excluded_lin():

	N = np.random.random_integers(4, 10, 1)  # doesn't matter
	K = 4  # this matters
	Y, D, X = random_data_base(N, K)
	propensity = propensity_wrapper(Y, D, X)

	assert_equal(propensity._get_excluded_lin([]), [0, 1, 2, 3])
	assert_equal(propensity._get_excluded_lin([3, 1]), [0, 2])
	assert_equal(propensity._get_excluded_lin([0, 1, 2, 3]), [])


def test_get_excluded_qua():

	Y, D, X = random_data()  # doesn't matter
	propensity = propensity_wrapper(Y, D, X)

	lin1 = [0, 2, 3]
	qua1 = [(0, 3), (3, 3)]
	ans1 = [(0, 0), (0, 2), (2, 2), (2, 3)]
	assert_equal(propensity._get_excluded_qua(lin1, qua1), ans1)

	lin2 = [1, 2]
	qua2 = []
	ans2 = [(1, 1), (1, 2), (2, 2)]
	assert_equal(propensity._get_excluded_qua(lin2, qua2), ans2)

	lin3 = [8, 5]
	qua3 = [(8, 8), (8, 5), (5, 5)]
	ans3 = []
	assert_equal(propensity._get_excluded_qua(lin3, qua3), ans3)


def test_calc_loglike():

	Y = random_data(D=False, X=False)  # shouldn't matter
	D = np.array([0, 0, 1, 1])  # this matters
	X = np.array([[1, 2], [3, 7], [1, 4], [3, 6]])  # this matters

	propensity = propensity_wrapper(Y, D, X)
	lin = [1]
	qua = [(0, 0)]
	ans = -2.567814
	assert np.allclose(propensity._calc_loglike(lin, qua), ans)


# constants used in helper functions
DEFAULT_N = 4
DEFAULT_K = 2
DEFAULT_LIN_B = []
DEFAULT_C_LIN = 0
DEFAULT_C_QUA = np.inf


# helper function
def random_data(Y=True, D=True, X=True):

	return random_data_base(DEFAULT_N, DEFAULT_K, Y, D, X)


# helper function
def propensity_wrapper(Y, D, X):

	return PropensitySelect(DEFAULT_LIN_B, DEFAULT_C_LIN, DEFAULT_C_QUA,
	                        CausalModel(Y, D, X))

