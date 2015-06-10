from nose.tools import *
import numpy as np

from ..causal import CausalModel
from ..core.propensity import Propensity

D = np.array([0, 0, 1, 1])
X = np.array([[1, 3], [5, 7], [8, 6], [4, 2]])
Y = np.random.rand(4)  # shouldn't matter

model = CausalModel(Y, D, X)
default_lin = 'all'
default_qua = []
propensity = Propensity(default_lin, default_qua, model)


def test_sigmoid():

	x = np.array([0, 10000, -10000, 5])
	ans = np.array([0.5, 1.0, 0.0, 1/(1+np.exp(-5))])
	assert np.array_equal(propensity._sigmoid(x), ans)


def test_log1exp():

	x = np.array([0, 10000, -10000, 5])
	ans = np.array([np.log(2), 0.0, 10000, np.log(1+np.exp(-5))])
	assert np.array_equal(propensity._log1exp(x), ans)


def test_neg_loglike():

	beta = np.array([1, 2])
	X_c = np.array([[100, 50], [-2, 1], [-500, -1300], [1, 0]])
	X_t = np.array([[0, 0], [50, 25], [-50, -75], [0, -0.5]])
	ans = 2 * (200 + np.log(2) + np.log(1+np.e))
	assert_equal(propensity._neg_loglike(beta, X_c, X_t), ans)


def test_form_matrix():

	ans0 = np.column_stack((np.ones(4), X))
	assert np.array_equal(propensity._form_matrix('all', []), ans0)

	lin1 = [0]
	qua1 = [(0, 1), (1, 1)]
	ans1 = np.array([[1, 1, 3, 9], [1, 5, 35, 49],
	                 [1, 8, 48, 36], [1, 4, 8, 4]])
	assert np.array_equal(propensity._form_matrix(lin1, qua1), ans1)

	lin2 = 'all'
	qua2 = [(0, 0)]
	ans2 = np.array([[1, 1, 3, 1], [1, 5, 7, 25],
	                 [1, 8, 6, 64], [1, 4, 2, 16]])
	assert np.array_equal(propensity._form_matrix(lin2, qua2), ans2)

