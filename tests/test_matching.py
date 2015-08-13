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

