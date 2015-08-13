from nose.tools import *
import numpy as np

import causalinference.core.data as d
import causalinference.core.propensity as p
from utils import random_data


def test_form_matrix():

	X = np.array([[1, 3], [5, 7], [8, 6], [4, 2]])

	ans0 = np.column_stack((np.ones(4), X))
	assert np.array_equal(p.form_matrix(X, [0, 1], []), ans0)

	lin1 = [0]
	qua1 = [(0, 1), (1, 1)]
	ans1 = np.array([[1, 1, 3, 9], [1, 5, 35, 49],
	                 [1, 8, 48, 36], [1, 4, 8, 4]])
	assert np.array_equal(p.form_matrix(X, lin1, qua1), ans1)

	lin2 = [0]
	qua2 = [(1, 0), (1, 1)]
	ans2 = np.array([[1, 1, 3, 9], [1, 5, 35, 49],
	                 [1, 8, 48, 36], [1, 4, 8, 4]])
	assert np.array_equal(p.form_matrix(X, lin2, qua2), ans2)

	lin3 = [0, 1]
	qua3 = [(0, 0)]
	ans3 = np.array([[1, 1, 3, 1], [1, 5, 7, 25],
	                 [1, 8, 6, 64], [1, 4, 2, 16]])
	assert np.array_equal(p.form_matrix(X, lin3, qua3), ans3)


def test_sigmoid():

	x = np.array([0, 10000, -10000, 5])
	ans = np.array([0.5, 1.0, 0.0, 1/(1+np.exp(-5))])
	assert np.array_equal(p.sigmoid(x), ans)


def test_log1exp():

	x = np.array([0, 10000, -10000, 5])
	ans = np.array([np.log(2), 0.0, 10000, np.log(1+np.exp(-5))])
	assert np.array_equal(p.log1exp(x), ans)


def test_neg_loglike():

	beta = np.array([1, 2])
	X_c = np.array([[100, 50], [-2, 1], [-500, -1300], [1, 0]])
	X_t = np.array([[0, 0], [50, 25], [-50, -75], [0, -0.5]])
	ans = 2 * (200 + np.log(2) + np.log(1+np.e))
	assert_equal(p.neg_loglike(beta, X_c, X_t), ans)


def test_neg_gradient():

	beta = np.array([2, -1])
	X_c = np.array([[1, 2], [125, 50]])
	X_t = np.array([[50, 0], [2.5, 4]])
	ans = np.array([125.5 - 2.5/(1+np.e), 51 - 4/(1+np.e)])
	assert np.array_equal(p.neg_gradient(beta, X_c, X_t), ans)


def test_calc_coef():

	X_c = np.array([[1, 1, 8], [1, 8, 5]])
	X_t = np.array([[1, 10, 2], [1, 5, 8]])
	ans = np.array([-6.9441137, 0.6608454, 0.4900669])

	assert np.allclose(p.calc_coef(X_c, X_t), ans)


def test_calc_se():

	Z = np.array([[1, 64, 188], [1, 132, 59], [1, 106, 72], [1, 86, 154]])
	phat = np.array([0.5101151, 0.3062871, 0.8566664, 0.3269315])
	ans = np.array([25.56301220, 0.16572624, 0.07956535])

	assert np.allclose(p.calc_se(Z, phat), ans)


def test_propensity():

	D = np.array([0, 0, 0, 1, 1, 1])
	X = np.array([[7, 8], [3, 10], [7, 10], [4, 7], [5, 10], [9, 8]])
	Y = random_data(D_cur=D, X_cur=X)

	data = d.Data(Y, D, X)
	propensity = p.Propensity(data, [0, 1], [])
	lin = [0, 1]
	qua = []
	coef = np.array([6.8066090, -0.0244874, -0.7524939])
	loglike = -3.626517
	fitted = np.array([0.6491366, 0.3117840, 0.2911631,
	                   0.8086407, 0.3013733, 0.6379023])
	se = np.array([8.5373779, 0.4595191, 0.8106499])
	keys = {'lin', 'qua', 'coef', 'loglike', 'fitted', 'se'}
	
	assert_equal(propensity['lin'], lin)
	assert_equal(propensity['qua'], qua)
	assert np.allclose(propensity['coef'], coef)
	assert np.allclose(propensity['loglike'], loglike)
	assert np.allclose(propensity['fitted'], fitted)
	assert np.allclose(propensity['se'], se)
	assert_equal(set(propensity.keys()), keys)

