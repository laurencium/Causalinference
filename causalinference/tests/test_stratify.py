from nose.tools import *
import numpy as np

from ..causal import CausalModel
from ..utils.simulate import SimulateData


def test_stratify():

	Y, D, X = SimulateData()
	causal = CausalModel(Y, D, X)
	causal.propensity()
	
	causal.stratify_s()
	assert_equal(len(causal.strata), len(causal.blocks)-1)
	assert_equal(sum([s.N for s in causal.strata]), causal.N)
	assert_equal(sum([s.N_t for s in causal.strata]), causal.N_t)

	causal.blocks = np.random.random_integers(1,10)
	causal.stratify()
	assert_equal(len(causal.strata), len(causal.blocks)-1)
	assert_equal(sum([s.N for s in causal.strata]), causal.N)
	assert_equal(sum([s.N_t for s in causal.strata]), causal.N_t)

