from nose.tools import *
import numpy as np

import causalinference.core.data as d
import causalinference.core.summary as s


def test_calc_ndiff():

	ans = -1/np.sqrt(2.5)
	assert_equal(s.calc_ndiff(4, 3, 2, 1), ans)


def test_summary():

	Y = np.array([1, 2, 3, 4])
	D = np.array([0, 0, 1, 1])
	X = np.array([[1, 3], [5, 7], [8, 6], [4, 2]])
	data = d.Data(Y, D, X)
	summary = s.Summary(data)

	Y_c_mean = 1.5
	Y_t_mean = 3.5
	Y_c_sd = 0.5
	Y_t_sd = 0.5
	rdiff = 2
	X_c_mean = np.array([3, 5])
	X_t_mean = np.array([6, 4])
	X_c_sd = np.sqrt([8, 8])
	X_t_sd = np.sqrt([8, 8])
	ndiff = np.array([3/(2*np.sqrt(2)), -1/(2*np.sqrt(2))])
	keys = set(['N_c', 'N_t', 'Y_c_mean', 'Y_t_mean', 'Y_c_sd', 'Y_t_sd',
	            'X_c_mean', 'X_t_mean', 'X_c_sd', 'X_t_sd',
		    'rdiff', 'ndiff'])

	assert_equal(summary['Y_c_mean'], Y_c_mean)
	assert_equal(summary['Y_t_mean'], Y_t_mean)
	assert_equal(summary['Y_c_sd'], Y_c_sd)
	assert_equal(summary['Y_t_sd'], Y_t_sd)
	assert_equal(summary['rdiff'], rdiff)
	assert np.array_equal(summary['X_c_mean'], X_c_mean)
	assert np.array_equal(summary['X_t_mean'], X_t_mean)
	assert np.array_equal(summary['X_c_sd'], X_c_sd)
	assert np.array_equal(summary['X_t_sd'], X_t_sd)
	assert np.array_equal(summary['ndiff'], ndiff)
	assert_equal(set(summary.keys()), keys)

