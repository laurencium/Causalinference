from nose.tools import *
import numpy as np

from ..causal import CausalModel
from ..core.covariates import Covariates

D = np.array([0, 0, 1, 1])
X = np.array([[1, 3], [5, 7], [8, 6], [4, 2]])
Y = np.random.rand(D.shape[0])  # shouldn't matter

model = CausalModel(Y, D, X)
covariates = Covariates(model)

def test_calc_means():

	mean_c = np.array([3, 5])
	mean_t = np.array([6, 4])
	assert np.array_equal(covariates._calc_means()[0], mean_c)
	assert np.array_equal(covariates._calc_means()[1], mean_t)


def test_calc_sds():

	sd = np.sqrt([8, 8])
	assert np.array_equal(covariates._calc_sds()[0], sd)
	assert np.array_equal(covariates._calc_sds()[1], sd)
	

def test_calc_ndiff():

	ans = -1/np.sqrt(2.5)
	assert_equal(covariates._calc_ndiff(4, 3, 2, 1), ans)


