from nose.tools import *
import numpy as np

import causalinference.estimators.ols as o


def test_add_const():

	X = np.array([[1, 2], [3, 4], [-2.4, 2]])
	ans = np.array([[1, 1, 2], [1, 3, 4], [1, -2.4, 2]])

	assert np.array_equal(o.add_const(X), ans)


def test_calc_te():

	Y_c = np.array([52, 30, 5])
	Y_t = np.array([12, 10, 44])
	X_c = np.array([[1], [3], [9]])
	X_t = np.array([[5], [4], [2]])
	ans1, ans2, ans3 = -12.684066, -10.65385, -14.71429

	ate, att, atc = o.calc_te(Y_c, Y_t, X_c, X_t)

	assert np.allclose(ate, ans1)
	assert np.allclose(att, ans2)
	assert np.allclose(atc, ans3)

