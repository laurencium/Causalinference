from nose.tools import *
import numpy as np

import causalinference.core.data as d
import causalinference.core.summary as s


def test_calc_ndiff():

	ans = -1/np.sqrt(2.5)
	assert_equal(s.calc_ndiff(4, 3, 2, 1), ans)


def test_summary():

	Y = np.array([1, 2, 3, 4, 6, 5])
	D = np.array([0, 0, 1, 1, 0, 1])
	X = np.array([[1, 3], [5, 7], [8, 6], [4, 2], [9, 11], [12, 10]])
	data = d.Data(Y, D, X)
	summary = s.Summary(data)

	N = 6
	K = 2
	N_c = 3
	N_t = 3
	Y_c_mean = 3
	Y_t_mean = 4
	Y_c_sd = np.sqrt(7)
	Y_t_sd = 1
	rdiff = 1
	X_c_mean = np.array([5, 7])
	X_t_mean = np.array([8, 6])
	X_c_sd = np.array([4, 4])
	X_t_sd = np.array([4, 4])
	ndiff = np.array([0.75, -0.25])
	keys1 = {'N', 'K', 'N_c', 'N_t', 'Y_c_mean', 'Y_t_mean', 'Y_c_sd', 'Y_t_sd',
	         'X_c_mean', 'X_t_mean', 'X_c_sd', 'X_t_sd', 'rdiff', 'ndiff'}

	assert_equal(summary['N'], N)
	assert_equal(summary['N_c'], N_c)
	assert_equal(summary['N_t'], N_t)
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
	assert_equal(set(summary.keys()), keys1)

	p_c = np.array([0.3, 0.5, 0.7])
	p_t = np.array([0.1, 0.5, 0.9])
	summary._summarize_pscore(p_c, p_t)
	keys2 = {'N', 'K', 'N_c', 'N_t', 'Y_c_mean', 'Y_t_mean', 'Y_c_sd', 'Y_t_sd',
	         'X_c_mean', 'X_t_mean', 'X_c_sd', 'X_t_sd', 'rdiff', 'ndiff',
		 'p_min', 'p_max', 'p_c_mean', 'p_t_mean'}
	
	assert_equal(summary['p_min'], 0.1)
	assert_equal(summary['p_max'], 0.9)
	assert_equal(summary['p_c_mean'], 0.5)
	assert_equal(summary['p_t_mean'], 0.5)
	assert_equal(set(summary.keys()), keys2)

