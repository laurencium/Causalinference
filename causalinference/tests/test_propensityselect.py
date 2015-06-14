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

	included1 = []
	ans1 = [0, 1, 2, 3]
	assert_equal(propensity._get_excluded_lin(included1), ans1)

	included2 = [3, 1]
	ans2 = [0, 2]
	assert_equal(propensity._get_excluded_lin(included2), ans2)

	included3 = [0, 1, 2, 3]
	ans3 = []
	assert_equal(propensity._get_excluded_lin(included3), ans3)


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


def test_test_lin():

	N = np.random.random_integers(4, 10, 1)  # doesn't matter
	K = 4  # this matters
	Y, D, X = random_data_base(N, K)  # data doesn't matter
	propensity = propensity_wrapper(Y, D, X)

	lin1 = [0, 1, 2, 3]
	C1 = np.random.rand(1)
	ans1 = [0, 1, 2, 3]
	assert_equal(propensity._test_lin(lin1, C1), ans1)

	Y = random_data(D=False, X=False)  # shouldn't matter
	D = np.array([0, 0, 1, 1])  # this matters
	X = np.array([[1, 2], [9, 7], [1, 4], [9, 6]])  # this matters
	propensity = propensity_wrapper(Y, D, X)

	lin2 = []
	C2 = 0.07
	ans2 = []
	assert_equal(propensity._test_lin(lin2, C2), ans2)

	lin3 = []
	C3 = 0.06
	ans3 = [1, 0]
	assert_equal(propensity._test_lin(lin3, C3), ans3)

	lin4 = [1]
	C4 = 0.35
	ans4 = [1]
	assert_equal(propensity._test_lin(lin4, C4), ans4)

	lin5 = [1]
	C5 = 0.34
	ans5 = [1, 0]
	assert_equal(propensity._test_lin(lin5, C5), ans5)


def test_select_lin_terms():

	N = np.random.random_integers(4, 10, 1)  # doesn't matter
	K = 4  # this matters
	Y, D, X = random_data_base(N, K)  # data doesn't matter
	propensity = propensity_wrapper(Y, D, X)

	lin1 = [3, 0, 1]
	C1 = np.inf
	ans1 = [3, 0, 1]
	assert_equal(propensity._select_lin_terms(lin1, C1), ans1)

	lin2 = [2]
	C2 = 0
	ans2 = [2, 0, 1, 3]
	assert_equal(propensity._select_lin_terms(lin2, C2), ans2)
	
	lin3 = []
	C3 = 0
	ans3 = [0, 1, 2, 3]
	assert_equal(propensity._select_lin_terms(lin3, C3), ans3)
	
	lin4 = [3, 1]
	C4 = -34.234
	ans4 = [3, 1, 0, 2]
	assert_equal(propensity._select_lin_terms(lin4, C4), ans4)

	Y = random_data(D=False, X=False)  # shouldn't matter
	D = np.array([0, 0, 1, 1])  # this matters
	X = np.array([[1, 2], [9, 7], [1, 4], [9, 6]])  # this matters
	propensity = propensity_wrapper(Y, D, X)

	lin5 = []
	C5 = 0.06
	ans5 = [1, 0]
	assert_equal(propensity._select_lin_terms(lin5, C5), ans5)


def test_test_qua():

	Y, D, X = random_data()  # doesn't matter
	propensity = propensity_wrapper(Y, D, X)

	lin1 = [1, 0]
	qua1 = [(1, 0), (0, 0), (1, 1)]
	C1 = np.random.rand(1)
	ans1 = [(1, 0), (0, 0), (1, 1)]
	assert_equal(propensity._test_qua(lin1, qua1, C1), ans1)

	lin2 = [1]
	qua2 = [(1, 1)]
	C2 = np.random.rand(1)
	ans2 = [(1, 1)]
	assert_equal(propensity._test_qua(lin2, qua2, C2), ans2)

	Y = random_data_base(N=6, K=2, D=False, X=False)  # doesn't matter
	D = np.array([0, 0, 0, 1, 1, 1])
	X = np.array([[7, 8], [3, 10], [7, 10], [4, 7], [5, 10], [9, 8]])
	propensity = propensity_wrapper(Y, D, X)

	lin3 = [0, 1]
	qua3 = []
	C3 = 1.2
	ans3 = []
	assert_equal(propensity._test_qua(lin3, qua3, C3), ans3)

	lin4 = [0, 1]
	qua4 = []
	C4 = 1.1
	ans4 = [(1, 1), (0, 1), (0, 0)]
	assert_equal(propensity._test_qua(lin4, qua4, C4), ans4)

	lin5 = [0, 1]
	qua5 = [(1, 1)]
	C5 = 2.4
	ans5 = [(1, 1)]
	assert_equal(propensity._test_qua(lin5, qua5, C5), ans5)

	lin6 = [0, 1]
	qua6 = [(1, 1)]
	C6 = 2.3
	ans6 = [(1, 1), (0, 1), (0, 0)]
	assert_equal(propensity._test_qua(lin6, qua6, C6), ans6)

	lin7 = [0, 1]
	qua7 = [(1, 1), (0, 1)]
	C7 = 3.9
	ans7 = [(1, 1), (0, 1)]
	assert_equal(propensity._test_qua(lin7, qua7, C7), ans7)

	lin8 = [0, 1]
	qua8 = [(1, 1), (0, 1)]
	C8 = 3.8
	ans8 = [(1, 1), (0, 1), (0, 0)]
	assert_equal(propensity._test_qua(lin8, qua8, C8), ans8)


def test_select_qua_terms():

	Y, D, X = random_data()  # doesn't matter
	propensity = propensity_wrapper(Y, D, X)

	lin1 = [0, 1]
	C1 = np.inf
	ans1 = []
	assert_equal(propensity._select_qua_terms(lin1, C1), ans1)

	lin2 = [1, 0]
	C2 = 0
	ans2 = [(1, 1), (1, 0), (0, 0)]
	assert_equal(propensity._select_qua_terms(lin2, C2), ans2)
	
	lin3 = [0]
	C3 = -983.340
	ans3 = [(0, 0)]
	assert_equal(propensity._select_qua_terms(lin3, C3), ans3)
	
	lin4 = []
	C4 = 34.234
	ans4 = []
	assert_equal(propensity._select_qua_terms(lin4, C4), ans4)

	Y = random_data_base(N=6, K=2, D=False, X=False)  # doesn't matter
	D = np.array([0, 0, 0, 1, 1, 1])
	X = np.array([[7, 8], [3, 10], [7, 10], [4, 7], [5, 10], [9, 8]])
	propensity = propensity_wrapper(Y, D, X)

	lin5 = [0, 1]
	qua5 = []
	C5 = 1.1
	ans5 = [(1, 1), (0, 1), (0, 0)]
	assert_equal(propensity._test_qua(lin5, qua5, C5), ans5)


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

