from nose.tools import *
import numpy as np

import causalinference.estimators.blocking as b
import causalinference.causal as c


def test_calc_atx():

	atxs = [0.5, 3.2, -9.4]
	Ns = [5, 13, 7]
	ans = -0.868

	assert np.allclose(b.calc_atx(atxs, Ns), ans)


def test_atx_se():

	atx_ses = [0.3, 1.3, 0.8]
	Ns = [3, 8, 4]
	ans = 0.72788888

	assert np.allclose(b.calc_atx_se(atx_ses, Ns), ans)


def test_blocking():

	Y1 = np.array([52, 30, 5, 29, 12, 10, 44, 87])
	D1 = np.array([0, 0, 0, 0, 1, 1, 1, 1])
	X1 = np.array([[1, 42], [3, 32], [9, 7], [12, 86],
	               [5, 94], [4, 36], [2, 13], [6, 61]])
	causal1 = c.CausalModel(Y1, D1, X1)
	Y2 = np.array([16, 4, 10, 6, 9, 11])
	D2 = np.array([0, 0, 0, 1, 1, 1])
	X2 = np.array([[1], [3], [3], [1], [7], [2]])
	causal2 = c.CausalModel(Y2, D2, X2)
	strata = [causal1, causal2]
	blocking = b.Blocking(strata)
	ate = 17.83044057
	atc = 35.45842407
	att = 0.20250793

	assert np.allclose(blocking['ate'], ate)
	assert np.allclose(blocking['atc'], atc)
	assert np.allclose(blocking['att'], att)

