from nose.tools import *
import numpy as np

import causalinference.estimators.ols as o
import causalinference.core.data as d


def test_add_const():

	X = np.array([[1, 2], [3, 4], [-2.4, 2]])
	ans = np.array([[1, 1, 2], [1, 3, 4], [1, -2.4, 2]])

	assert np.array_equal(o.add_const(X), ans)


def test_calc_atc():

	Y_c = np.array([52, 30, 5, 29])
	Y_t = np.array([12, 10, 44, 87])
	X_c = np.array([[1, 42], [3, 32], [9, 7], [12, 86]])
	X_t = np.array([[5, 94], [4, 36], [2, 13], [6, 61]])
	ans = 63.2095

	assert np.allclose(o.calc_atc(Y_c, Y_t, X_c, X_t), ans)


def test_calc_att():

	Y_c = np.array([52, 30, 5, 29])
	Y_t = np.array([12, 10, 44, 87])
	X_c = np.array([[1, 42], [3, 32], [9, 7], [12, 86]])
	X_t = np.array([[5, 94], [4, 36], [2, 13], [6, 61]])
	ans = -2.020611

	assert np.allclose(o.calc_att(Y_c, Y_t, X_c, X_t), ans)


def test_calc_ate():

	atc = 63.2095
	att = -2.020611
	N_c = 4
	N_t = 4
	ans = 30.59444

	assert np.allclose(o.calc_ate(atc, att, N_c, N_t), ans)


def test_ols():

	Y = np.array([52, 30, 5, 29, 12, 10, 44, 87])
	D = np.array([0, 0, 0, 0, 1, 1, 1, 1])
	X = np.array([[1, 42], [3, 32], [9, 7], [12, 86],
	              [5, 94], [4, 36], [2, 13], [6, 61]])
	data = d.Data(Y, D, X)
	ols = o.OLS(data)
	atc = 63.2095
	att = -2.020611
	ate = 30.59444
	keys = {'atc', 'att', 'ate'}

	assert np.allclose(ols['atc'], atc)
	assert np.allclose(ols['att'], att)
	assert np.allclose(ols['ate'], ate)
	assert_equal(set(ols.keys()), keys)

