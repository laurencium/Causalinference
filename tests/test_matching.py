from __future__ import division
from nose.tools import *
import numpy as np

import causalinference.estimators.matching as m


def test_norm():

	X_i = np.array([1, 7, 3])
	X_m = np.array([[4, 2, 5], [9, 8, 6]])

	W1 = np.array([0.5, 1, 0.25])
	ans1 = np.array([30.5, 35.25])
	assert np.array_equal(m.norm(X_i, X_m, W1), ans1)

	W2 = np.array([[0.5, -0.1, 0.7], [-0.1, 1, 3], [0.7, 3, 0.25]])
	ans2 = np.array([-18.1, 85.25])
	assert np.array_equal(m.norm(X_i, X_m, W2), ans2)


def test_smallestm():

	d1 = np.array([1, 3, 2])
	m1 = 1
	ans1 = np.array([0])
	assert_equal(set(m.smallestm(d1, m1)), set(ans1))

	d2 = np.array([1, 3, 2])
	m2 = 2
	ans2 = np.array([0, 2])
	assert_equal(set(m.smallestm(d2, m2)), set(ans2))

	d3 = np.array([9, 2, 5, 9, 1, 2, 7])
	m3 = 1
	ans3 = np.array([4])
	assert_equal(set(m.smallestm(d3, m3)), set(ans3))

	d4 = np.array([9, 2, 5, 9, 1, 2, 7])
	m4 = 2
	ans4 = np.array([4, 1, 5])
	assert_equal(set(m.smallestm(d4, m4)), set(ans4))

	d5 = np.array([9, 2, 5, 9, 1, 2, 7])
	m5 = 3
	ans5 = np.array([4, 1, 5])
	assert_equal(set(m.smallestm(d5, m5)), set(ans5))

	d6 = np.array([9, 2, 5, 9, 1, 2, 7])
	m6 = 4
	ans6 = np.array([4, 1, 5, 2])
	assert_equal(set(m.smallestm(d6, m6)), set(ans6))

	d7 = np.array([-3.2, -3.2, 9.66, -3.2, 28.4])
	m7 = 1
	ans7 = np.array([0, 1, 3])
	assert_equal(set(m.smallestm(d7, m7)), set(ans7))


def test_match():

	X_i = np.array([1, 7, 3])
	X_m = np.array([[9, 8, 6], [4, 2, 5]])

	W1 = np.array([0.5, 1, 0.25])
	m1 = 1
	ans1 = np.array([1])
	assert_equal(set(m.match(X_i, X_m, W1, m1)), set(ans1))

	W2 = np.array([[0.5, -0.1, 0.7], [-0.1, 1, 3], [0.7, 3, 0.25]])
	m2 = 1
	ans2 = np.array([1])
	assert_equal(set(m.match(X_i, X_m, W2, m2)), set(ans2))


def test_bias_coefs():

	Y_m = np.array([4, 2, 5, 2])
	X_m = np.array([[7, 6], [5, 4], [2, 3], [3, 5]])
	matches = [np.array([1, 0, 2]), np.array([1, 2]),
	           np.array([2, 0]), np.array([0]), np.array([0, 1])]

	ans = np.array([-2, 3])
	assert np.allclose(m.bias_coefs(matches, Y_m, X_m), ans)


def test_bias():

	X = np.array([[1, 2, 3], [-3, -2, -1]])
	X_m = np.array([[4, 2, 6], [5, 7, 3], [9, 4, 1]])
	matches = [np.array([0, 1, 2]), np.array([1])]
	coefs = np.array([-2, 0, 3])

	ans = np.array([-9, -4])
	assert np.allclose(m.bias(X, X_m, matches, coefs), ans)


def test_scaled_counts():

	N = 10
	matches = [np.array([3, 0, 1]), np.array([7]), np.array([1, 9])]

	ans = np.array([1/3, 1/3+1/2, 0, 1/3, 0, 0, 0, 1, 0, 1/2])
	assert np.allclose(m.scaled_counts(N, matches), ans)


def test_calc_atx_var():

	vars_c = np.array([1, 2])
	vars_t = np.array([0.5, 1, 0.25])
	weights_c = np.array([1.5, 0.5])
	weights_t = np.array([1, 1, 1])

	out_var = m.calc_atx_var(vars_c, vars_t, weights_c, weights_t)
	ans = 0.8819444
	assert np.allclose(out_var, ans)
	

def test_calc_atc_se():

	vars_c = np.array([1, 2])
	vars_t = np.array([0.5, 1, 0.25])
	scaled_counts_t = np.array([1, 1, 0])

	out_se = m.calc_atc_se(vars_c, vars_t, scaled_counts_t)
	ans = 1.0606602
	assert np.allclose(out_se, ans)


def test_calc_att_se():

	vars_c = np.array([1, 2])
	vars_t = np.array([0.5, 1, 0.25])
	scaled_counts_c = np.array([1, 2])

	out_se = m.calc_att_se(vars_c, vars_t, scaled_counts_c)
	ans = 1.0929064
	assert np.allclose(out_se, ans)


def test_calc_ate_se():

	vars_c = np.array([1, 2])
	vars_t = np.array([0.5, 1, 0.25])
	scaled_counts_c = np.array([1, 2])
	scaled_counts_t = np.array([1, 1, 0])

	out_se = m.calc_ate_se(vars_c, vars_t, scaled_counts_c, scaled_counts_t)
	ans = 1.0630146
	assert np.allclose(out_se, ans)

