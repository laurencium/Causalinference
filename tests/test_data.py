from nose.tools import *
import numpy as np

import causalinference.core.data as d


def test_preprocess():

	Y1 = np.array([[1.2], [3.45], [-6], [78.90]])
	D1 = np.array([[0], [1], [0.0], [1]])
	X1 = np.array([-1, 3, -5.6, 8.9])
	Y_out, D_out, X_out = d.preprocess(Y1, D1, X1)

	ans1 = np.array([1.2, 3.45, -6, 78.9])
	assert np.array_equal(Y_out, ans1)

	ans2 = np.array([0, 1.0, -0.0, 1])
	assert np.array_equal(D_out, ans2)

	ans3 = np.array([[-1], [3], [-5.6], [8.9]])
	assert np.array_equal(X_out, ans3)


	Y2 = np.array([3, 98])
	D2 = np.array([[5], [21.9], [-53]])
	X2 = np.array([1, 3.14])
	assert_raises(IndexError, d.preprocess, Y2, D2, X2)


def test_data():

	Y1 = np.array([1.2, 3.45, -6, 78.90, -9, 8.7654])
	D1 = np.array([0, 1, 0, 1.0, 0.0, 1])
	X1 = np.array([[-1, 2], [3, -4], [-5.6, -7], [8.9, 0.0], [99, 877], [-666, 54321]])
	data = d.Data(Y1, D1, X1)

	ans1 = np.array([1.2, 3.45, -6, 78.9, -9, 8.7654])
	assert np.array_equal(data['Y'], ans1)

	ans2 = np.array([0, 1, 0, 1, 0, 1])
	assert np.array_equal(data['D'], ans2)

	ans3 = np.array([[-1, 2], [3, -4], [-5.6, -7], [8.9, 0], [99, 877], [-666, 54321]])
	assert np.array_equal(data['X'], ans3)

	ans4 = 6
	assert_equal(data['N'], ans4)

	ans5 = 2
	assert_equal(data['K'], ans5)

	ans6 = np.array([True, False, True, False, True, False])
	assert np.array_equal(data['controls'], ans6)

	ans7 = np.array([False, True, False, True, False, True])
	assert np.array_equal(data['treated'], ans7)

	ans8 = np.array([1.2, -6, -9])
	assert np.array_equal(data['Y_c'], ans8)

	ans9 = np.array([3.45, 78.9, 8.7654])
	assert np.array_equal(data['Y_t'], ans9)

	ans10 = np.array([[-1, 2], [-5.6, -7], [99, 877]])
	assert np.array_equal(data['X_c'], ans10)

	ans11 = np.array([[3, -4], [8.9, 0], [-666, 54321]])
	assert np.array_equal(data['X_t'], ans11)

	ans12 = 3
	assert_equal(data['N_t'], ans12)

	ans13 = 3
	assert_equal(data['N_c'], ans13)

	ans14 = 'int'
	assert_equal(data['D'].dtype, ans14)

	ans15 = {'Y', 'D', 'X', 'N', 'K', 'controls', 'treated',
	         'Y_c', 'Y_t', 'X_c', 'X_t', 'N_c', 'N_t'}
	assert_equal(set(data.keys()), ans15)

	Y2 = np.array([[1.2], [3.45], [-6], [78.90]])
	D2 = np.array([[0], [1], [0.0], [1]])
	X2 = np.array([[-1, 2], [3, -4], [-5.6, -7], [8.9, 0.0]])
	assert_raises(ValueError, d.Data, Y2, D2, X2)

