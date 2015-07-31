from nose.tools import *
import numpy as np

import causalinference.estimators.ols as o
import causalinference.core.data as d


def test_form_matrix():

	D = np.array([0, 1, 0, 1])
	X = np.array([[1], [2], [3], [4]])

	adj1 = 0
	ans1 = np.array([[1, 0], [1, 1], [1, 0], [1, 1]])
	assert np.array_equal(o.form_matrix(D, X, adj1), ans1)

	adj2 = 1
	ans2 = np.array([[1, 0, -1.5], [1, 1, -0.5],
	                 [1, 0, 0.5], [1, 1, 1.5]])
	assert np.array_equal(o.form_matrix(D, X, adj2), ans2)

	adj3 = 2
	ans3 = np.array([[1, 0, -1.5, 0], [1, 1, -0.5, -0.5],
	                [1, 0, 0.5, 0], [1, 1, 1.5, 1.5]])
	assert np.array_equal(o.form_matrix(D, X, adj3), ans3)


def test_calc_ate():

	olscoef = np.array([1, 2, 3, 4])
	ans = 2

	assert_equal(o.calc_ate(olscoef), ans)


def test_calc_atx():

	olscoef = np.array([1, 2, 3, 4, 5, 6])
	meandiff = np.array([7, 8])
	ans = 85

	assert_equal(o.calc_atx(olscoef, meandiff), ans)


def test_calc_cov():

	Z = np.array([[4, 4, 4, 2, 1, 3], [4, 2, 2, 6, 2, 2],
	              [3, 4, 2, 1, 3, 1], [2, 3, 0, 0, 1, 2],
		      [4, 3, 2, 1, 4, 2], [2, 5, 4, 2, 2, 0]])
	u = np.array([1, 3, 6, 4, 3, 1])
	ans = np.array([[434.755102, 8.442177, -87.529252,
	                 -77.227211, -204.360544, -354.38095],
			[8.442177, 1.988662, -3.601814,
			 -1.224943, -4.913832, -6.68254],
			[-87.529252, -3.601814, 19.817710,
			 15.136009, 41.933787, 71.05079],
			[-77.227211, -1.224943, 15.136009,
			 14.185125, 35.989569, 62.89841],
			[-204.360544, -4.913831, 41.933787,
			 35.989569, 97.145125, 166.58730],
			[-354.380952, -6.682540, 71.050794,
			 62.898413, 166.587302, 289.11111]])

	assert np.allclose(o.calc_cov(Z, u), ans)


def test_submatrix():

	cov = np.array([[1, 2, 3, 4, 5, 6], [7, 9, 8, 9, 8, 7],
	                [1, 2, 3, 4, 5, 6], [7, 8, 9, 1, 2, 3],
			[4, 6, 5, 6, 5, 4], [7, 3, 8, 9, 2, 1]])
	ans = np.array([[9, 8, 7], [6, 5, 4], [3, 2, 1]])

	assert np.allclose(o.submatrix(cov), ans)


def test_calc_ate_se():

	subcov = np.array([[9, 8, 7], [6, 5, 4], [3, 2, 1]])
	ans = np.sqrt(5)

	assert_equal(o.calc_ate_se(subcov), ans)


def test_calc_atx_se():

	cov = np.array([[1, 2, 3, 4, 5, 6], [7, 9, 8, 9, 8, 7],
	                [1, 2, 3, 4, 5, 6], [7, 8, 9, 1, 2, 3],
			[4, 6, 5, 6, 5, 4], [7, 3, 8, 9, 2, 1]])
	meandiff = np.array([3, 7])
	ans = 18.46619

	assert np.allclose(o.calc_atx_se(cov, meandiff), ans)


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
	keys = {'ate', 'atc', 'att', 'ate_se', 'atc_se', 'att_se'}

	assert np.allclose(ols['atc'], atc)
	assert np.allclose(ols['att'], att)
	assert np.allclose(ols['ate'], ate)
	assert_equal(set(ols.keys()), keys)

