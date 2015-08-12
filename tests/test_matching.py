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

