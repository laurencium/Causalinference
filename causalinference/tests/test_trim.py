from nose.tools import *
import numpy as np

from ..causal import CausalModel
from ..utils.simulate import SimulateData


def test_trim():

	Y, D, X = SimulateData()
	N, K = X.shape
	N_t = D.sum()
	causal = CausalModel(Y, D, X)
	causal.propensity()
	
	causal.trim_s(True)
	trimmed = (causal.pscore['fitted'] < causal.cutoff) | \
	          (causal.pscore['fitted'] > 1-causal.cutoff)
	causal.trim()
	assert_equal(N-trimmed.sum(), causal.N)
	assert_equal(N_t-D[trimmed].sum(), causal.N_t)

