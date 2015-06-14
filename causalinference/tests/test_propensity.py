from nose.tools import *
import numpy as np

from ..causal import CausalModel
from ..core.propensity import Propensity
from tools import random_data


def test_form_matrix():

	X = np.array([[1, 3], [5, 7], [8, 6], [4, 2]])  # this matters
	Y, D = random_data(X_cur=X)
	propensity = propensity_wrapper(Y, D, X)

	ans0 = np.column_stack((np.ones(4), X))
	assert np.array_equal(propensity._form_matrix('all', []), ans0)

	lin1 = [0]
	qua1 = [(0, 1), (1, 1)]
	ans1 = np.array([[1, 1, 3, 9], [1, 5, 35, 49],
	                 [1, 8, 48, 36], [1, 4, 8, 4]])
	assert np.array_equal(propensity._form_matrix(lin1, qua1), ans1)

	lin2 = [0]
	qua2 = [(1, 0), (1, 1)]
	ans2 = np.array([[1, 1, 3, 9], [1, 5, 35, 49],
	                 [1, 8, 48, 36], [1, 4, 8, 4]])
	assert np.array_equal(propensity._form_matrix(lin2, qua2), ans2)

	lin3 = 'all'
	qua3 = [(0, 0)]
	ans3 = np.array([[1, 1, 3, 1], [1, 5, 7, 25],
	                 [1, 8, 6, 64], [1, 4, 2, 16]])
	assert np.array_equal(propensity._form_matrix(lin3, qua3), ans3)


def test_sigmoid():

	Y, D, X = random_data()
	propensity = propensity_wrapper(Y, D, X)

	x = np.array([0, 10000, -10000, 5])
	ans = np.array([0.5, 1.0, 0.0, 1/(1+np.exp(-5))])
	assert np.array_equal(propensity._sigmoid(x), ans)


def test_log1exp():

	Y, D, X = random_data()
	propensity = propensity_wrapper(Y, D, X)

	x = np.array([0, 10000, -10000, 5])
	ans = np.array([np.log(2), 0.0, 10000, np.log(1+np.exp(-5))])
	assert np.array_equal(propensity._log1exp(x), ans)


def test_neg_loglike():

	Y, D, X = random_data()
	propensity = propensity_wrapper(Y, D, X)

	beta = np.array([1, 2])
	X_c = np.array([[100, 50], [-2, 1], [-500, -1300], [1, 0]])
	X_t = np.array([[0, 0], [50, 25], [-50, -75], [0, -0.5]])
	ans = 2 * (200 + np.log(2) + np.log(1+np.e))
	assert_equal(propensity._neg_loglike(beta, X_c, X_t), ans)


def test_neg_gradient():

	Y, D, X = random_data()
	propensity = propensity_wrapper(Y, D, X)

	beta = np.array([2, -1])
	X_c = np.array([[1, 2], [125, 50]])
	X_t = np.array([[50, 0], [2.5, 4]])
	ans = np.array([125.5 - 2.5/(1+np.e), 51 - 4/(1+np.e)])
	assert np.array_equal(propensity._neg_gradient(beta, X_c, X_t), ans)


def test_calc_coef():

	Y, D, X = random_data()
	propensity = propensity_wrapper(Y, D, X)

	X_c = np.array([[1, 1, 8], [1, 8, 5]])
	X_t = np.array([[1, 10, 2], [1, 5, 8]])
	ans = np.array([-6.9441137, 0.6608454, 0.4900669])

	assert np.allclose(propensity._calc_coef(X_c, X_t), ans)


def test_calc_se():

	Y, D, X = random_data()
	propensity = propensity_wrapper(Y, D, X)

	Z = np.array([[1, 64, 188], [1, 132, 59], [1, 106, 72], [1, 86, 154]])
	p = np.array([0.5101151, 0.3062871, 0.8566664, 0.3269315])
	ans = np.array([25.56301220, 0.16572624, 0.07956535])

	assert np.allclose(propensity._calc_se(Z, p), ans)


def test_propensity():

	D = np.array([0, 0, 1, 1])
	X = np.array([[1, 2], [9, 7], [1, 4], [9, 6]])
	Y = random_data(D_cur=D, X_cur=X)

	model = CausalModel(Y, D, X)
	propensity = Propensity('all', [], model)
	coef = np.array([-2.1505403, -0.3671654, 0.8392352])
	loglike = -2.567814
	fitted = np.array([0.3016959, 0.6033917, 0.6983041, 0.3966083])
	se = np.array([3.8953529, 0.6507885, 1.3595614])
	keys = set(['lin', 'qua', 'coef', 'loglike', 'fitted', 'se'])
	
	assert_equal(propensity['lin'], 'all')
	assert_equal(propensity['qua'], [])
	assert np.allclose(propensity['coef'], coef)
	assert np.allclose(propensity['loglike'], loglike)
	assert np.allclose(propensity['fitted'], fitted)
	assert np.allclose(propensity['se'], se)
	assert_equal(set(propensity.keys()), keys)


# constants used in helper functions
DEFAULT_LIN = 'all'
DEFAULT_QUA = []


# helper function
def propensity_wrapper(Y, D, X):

	return Propensity(DEFAULT_LIN, DEFAULT_QUA, CausalModel(Y, D, X))

