from nose.tools import *
import numpy as np

import causalinference.estimators.weighting as w
import causalinference.core.data as d


def test_calc_weights():

	pscore = np.array([0.1, 0.25, 0.5, 0.75, 0.9])
	D = np.array([0, 1, 0, 1, 0])

	ans = np.array([1.11111, 4, 2, 1.33333, 10])
	assert np.allclose(w.calc_weights(pscore, D), ans)


def test_weigh_data():

	Y = np.array([1, -2, 3, -5, 7])
	D = np.array([0, 1, 0, 1, 0])
	X = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]])
	weights = np.array([1/0.9, 4, 2, 1/0.75, 10])

	Y_ans = np.array([1.11111, -8, 6, -6.66667, 70])
	Z_ans = np.array([[1.11111, 0, 1.11111, 2.22222],
	                [4, 4, 12, 16],
			[2, 0, 10, 12],
			[1.33333, 1.33333, 9.33333, 10.66667],
			[10, 0, 90, 100]])
	Y_out, Z_out = w.weigh_data(Y, D, X, weights)
	assert np.allclose(Y_out, Y_ans)
	assert np.allclose(Z_out, Z_ans)


def test_weighting():

	Y = np.array([1, -2, 3, -5, 7])
	D = np.array([0, 1, 0, 1, 0])
	X = np.array([3, 2, 3, 5, 5])
	pscore = np.array([0.1, 0.25, 0.5, 0.75, 0.9])
	data = d.Data(Y, D, X)
	data._dict['pscore'] = pscore

	weighting = w.Weighting(data)
	ate = -6.7963178
	ate_se = 2.8125913
	keys = {'ate', 'ate_se'}
	assert np.allclose(weighting['ate'], ate)
	assert np.allclose(weighting['ate_se'], ate_se)
	assert_equal(set(weighting.keys()), keys)

