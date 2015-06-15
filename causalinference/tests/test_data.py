from nose.tools import *
import numpy as np

from ..core.data import Data


def test_data():

	Y = np.array([1.2, 3.45, -6, 78.90])
	D = np.array([0, 1, 0, 1.0])
	X = np.array([[-1, 2], [3, -4], [-5.6, -7], [8.9, 0.0]])
	data = Data(Y, D, X)

	ans1 = np.array([1.2, 3.45, -6, 78.9])
	assert np.array_equal(data['Y'], ans1)

	ans2 = np.array([0, 1, 0, 1])
	assert np.array_equal(data['D'], ans2)

	ans3 = np.array([[-1, 2], [3, -4], [-5.6, -7], [8.9, 0]])
	assert np.array_equal(data['X'], ans3)

	ans4 = 4
	assert_equal(data['N'], ans4)

	ans5 = 2
	assert_equal(data['K'], ans5)

	ans6 = np.array([True, False, True, False])
	assert np.array_equal(data['controls'], ans6)

	ans7 = np.array([False, True, False, True])
	assert np.array_equal(data['treated'], ans7)

	ans8 = np.array([1.2, -6])
	assert np.array_equal(data['Y_c'], ans8)

	ans9 = np.array([3.45, 78.9])
	assert np.array_equal(data['Y_t'], ans9)

	ans10 = np.array([[-1, 2], [-5.6, -7]])
	assert np.array_equal(data['X_c'], ans10)

	ans11 = np.array([[3, -4], [8.9, 0]])
	assert np.array_equal(data['X_t'], ans11)

	ans12 = 2
	assert_equal(data['N_t'], ans12)

	ans13 = 2
	assert_equal(data['N_c'], ans13)

	ans14 = 'int'
	assert_equal(data['D'].dtype, ans14)

	ans15 = set(['Y', 'D', 'X', 'N', 'K', 'controls', 'treated',
	             'Y_c', 'Y_t', 'X_c', 'X_t', 'N_c', 'N_t'])
	assert_equal(set(data.keys()), ans15)

