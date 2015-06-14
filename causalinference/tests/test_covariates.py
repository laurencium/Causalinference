from nose.tools import *
import numpy as np

from ..causal import CausalModel
from ..core.covariates import Covariates
from tools import random_data


def test_calc_ndiff():

	Y, D, X = random_data()
	covariates = covariates_wrapper(Y, D, X)

	ans = -1/np.sqrt(2.5)
	assert_equal(covariates._calc_ndiff(4, 3, 2, 1), ans)


def test_covariates():

	D = np.array([0, 0, 1, 1])
	X = np.array([[1, 3], [5, 7], [8, 6], [4, 2]])
	Y = random_data(D_cur=D, X_cur=X)
	covariates = covariates_wrapper(Y, D, X)

	mean_c = np.array([3, 5])
	mean_t = np.array([6, 4])
	sd_c = np.sqrt([8, 8])
	sd_t = np.sqrt([8, 8])
	ndiff = np.array([3/(2*np.sqrt(2)), -1/(2*np.sqrt(2))])
	keys = set(['mean_c', 'mean_t', 'sd_c', 'sd_t', 'ndiff'])

	assert np.array_equal(covariates['mean_c'], mean_c)
	assert np.array_equal(covariates['mean_t'], mean_t)
	assert np.array_equal(covariates['sd_c'], sd_c)
	assert np.array_equal(covariates['sd_t'], sd_t)
	assert np.array_equal(covariates['ndiff'], ndiff)
	assert_equal(set(covariates.keys()), keys)


# help function
def covariates_wrapper(Y, D, X):

	return Covariates(CausalModel(Y, D, X))

