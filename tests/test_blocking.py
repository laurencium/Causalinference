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

	adj1 = 0
	blocking1 = b.Blocking(strata, adj1)
	ate1 = 4.714286
	atc1 = 4.714286
	att1 = 4.714286
	ate_se1 = 10.18945
	atc_se1 = 10.18945
	att_se1 = 10.18945
	assert np.allclose(blocking1['ate'], ate1)
	assert np.allclose(blocking1['atc'], atc1)
	assert np.allclose(blocking1['att'], att1)
	assert np.allclose(blocking1['ate_se'], ate_se1)
	assert np.allclose(blocking1['atc_se'], atc_se1)
	assert np.allclose(blocking1['att_se'], att_se1)

	adj2 = 1
	blocking2 = b.Blocking(strata, adj2)
	ate2 = 1.657703
	atc2 = 1.657703
	att2 = 1.657703
	ate_se2 = 10.22921
	atc_se2 = 10.22921
	att_se2 = 10.22921
	assert np.allclose(blocking2['ate'], ate2)
	assert np.allclose(blocking2['atc'], atc2)
	assert np.allclose(blocking2['att'], att2)
	assert np.allclose(blocking2['ate_se'], ate_se2)
	assert np.allclose(blocking2['atc_se'], atc_se2)
	assert np.allclose(blocking2['att_se'], att_se2)

	adj3 = 2
	blocking3 = b.Blocking(strata, adj3)
	ate3 = 17.83044057
	atc3 = 35.45842407
	att3 = 0.20250793
	ate_se3 = 11.42591
	atc_se3 = 17.11964
	att_se3 = 6.87632
	assert np.allclose(blocking3['ate'], ate3)
	assert np.allclose(blocking3['atc'], atc3)
	assert np.allclose(blocking3['att'], att3)
	assert np.allclose(blocking3['ate_se'], ate_se3)
	assert np.allclose(blocking3['atc_se'], atc_se3)
	assert np.allclose(blocking3['att_se'], att_se3)

