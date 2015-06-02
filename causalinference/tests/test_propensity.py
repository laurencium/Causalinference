from nose.tools import *
import numpy as np

from ..causal import CausalModel
from ..core.propensity import Propensity

D = np.array([0, 0, 1, 1])
X = np.array([[1, 3], [5, 7], [8, 6], [4, 2]])
Y = np.random.rand(D.shape[0])  # shouldn't matter

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

