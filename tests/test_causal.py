from __future__ import division
from nose.tools import *
import numpy as np

import causalinference.causal as c
from utils import random_data


def test_est_propensity():

	D = np.array([0, 0, 0, 1, 1, 1])
	X = np.array([[7, 8], [3, 10], [7, 10], [4, 7], [5, 10], [9, 8]])
	Y = random_data(D_cur=D, X_cur=X)
	causal = c.CausalModel(Y, D, X)

	causal.est_propensity()
	lin = [0, 1]
	qua = []
	coef = np.array([6.8066090, -0.0244874, -0.7524939])
	loglike = -3.626517
	fitted = np.array([0.6491366, 0.3117840, 0.2911631,
	                   0.8086407, 0.3013733, 0.6379023])
	se = np.array([8.5373779, 0.4595191, 0.8106499])
	keys = {'lin', 'qua', 'coef', 'loglike', 'fitted', 'se'}
	
	assert_equal(causal.propensity['lin'], lin)
	assert_equal(causal.propensity['qua'], qua)
	assert np.allclose(causal.propensity['coef'], coef)
	assert np.allclose(causal.propensity['loglike'], loglike)
	assert np.allclose(causal.propensity['fitted'], fitted)
	assert np.allclose(causal.propensity['se'], se)
	assert_equal(set(causal.propensity.keys()), keys)
	assert np.allclose(causal.raw_data['pscore'], fitted)
	

def test_est_propensity_s():

	D = np.array([0, 0, 0, 1, 1, 1])
	X = np.array([[7, 8], [3, 10], [7, 10], [4, 7], [5, 10], [9, 8]])
	Y = random_data(D_cur=D, X_cur=X)
	causal = c.CausalModel(Y, D, X)

	causal.est_propensity_s()
	lin1 = [1]
	qua1 = []
	coef1 = np.array([6.5424027, -0.7392041])
	loglike1 = -3.627939
	fitted1 = np.array([0.6522105, 0.2995088, 0.2995088,
	                   0.7970526, 0.2995088, 0.6522105])
	se1 = np.array([6.8455179, 0.7641445])
	keys = {'lin', 'qua', 'coef', 'loglike', 'fitted', 'se'}
	
	assert_equal(causal.propensity['lin'], lin1)
	assert_equal(causal.propensity['qua'], qua1)
	assert np.allclose(causal.propensity['coef'], coef1)
	assert np.allclose(causal.propensity['loglike'], loglike1)
	assert np.allclose(causal.propensity['fitted'], fitted1)
	assert np.allclose(causal.propensity['se'], se1)
	assert_equal(set(causal.propensity.keys()), keys)
	assert np.allclose(causal.raw_data['pscore'], fitted1)

	causal.est_propensity_s([0,1])
	lin2 = [0, 1]
	qua2 = []
	coef2 = np.array([6.8066090, -0.0244874, -0.7524939])
	loglike2 = -3.626517
	fitted2 = np.array([0.6491366, 0.3117840, 0.2911631,
	                    0.8086407, 0.3013733, 0.6379023])
	se2 = np.array([8.5373779, 0.4595191, 0.8106499])

	assert_equal(causal.propensity['lin'], lin2)
	assert_equal(causal.propensity['qua'], qua2)
	assert np.allclose(causal.propensity['coef'], coef2)
	assert np.allclose(causal.propensity['loglike'], loglike2)
	assert np.allclose(causal.propensity['fitted'], fitted2)
	assert np.allclose(causal.propensity['se'], se2)
	assert np.allclose(causal.raw_data['pscore'], fitted2)


def test_est_via_ols():

	Y = np.array([52, 30, 5, 29, 12, 10, 44, 87])
	D = np.array([0, 0, 0, 0, 1, 1, 1, 1])
	X = np.array([[1, 42], [3, 32], [9, 7], [12, 86],
	              [5, 94], [4, 36], [2, 13], [6, 61]])
	causal = c.CausalModel(Y, D, X)

	adj1 = 0
	causal.est_via_ols(adj1)
	ate1 = 9.25
	ate_se1 = 17.68253
	keys1 = {'ate', 'ate_se'}
	assert np.allclose(causal.estimates['ols']['ate'], ate1)
	assert np.allclose(causal.estimates['ols']['ate_se'], ate_se1)
	assert_equal(set(causal.estimates['ols'].keys()), keys1)

	adj2 = 1
	causal.est_via_ols(adj2)
	ate2 = 3.654552
	ate_se2 = 17.749993
	keys2 = {'ate', 'ate_se'}
	assert np.allclose(causal.estimates['ols']['ate'], ate2)
	assert np.allclose(causal.estimates['ols']['ate_se'], ate_se2)
	assert_equal(set(causal.estimates['ols'].keys()), keys2)

	adj3 = 2
	causal.est_via_ols(adj3)
	ate3 = 30.59444
	atc3 = 63.2095
	att3 = -2.020611
	ate_se3 = 19.91887865
	atc_se3 = 29.92152
	att_se3 = 11.8586
	keys3 = {'ate', 'atc', 'att', 'ate_se', 'atc_se', 'att_se'}
	assert np.allclose(causal.estimates['ols']['ate'], ate3)
	assert np.allclose(causal.estimates['ols']['atc'], atc3)
	assert np.allclose(causal.estimates['ols']['att'], att3)
	assert np.allclose(causal.estimates['ols']['ate_se'], ate_se3)
	assert np.allclose(causal.estimates['ols']['atc_se'], atc_se3)
	assert np.allclose(causal.estimates['ols']['att_se'], att_se3)
	assert_equal(set(causal.estimates['ols'].keys()), keys3)


def test_parse_lin_terms():

	K1 = 4
	lin1 = None
	ans1 = []
	assert_equal(c.parse_lin_terms(K1, lin1), ans1)

	K2 = 2
	lin2 = 'all'
	ans2 = [0, 1]
	assert_equal(c.parse_lin_terms(K2, lin2), ans2)

	K3 = 2
	lin3 = [1]
	ans3 = [1]
	assert_equal(c.parse_lin_terms(K3, lin3), ans3)

	K4 = 2
	lin4 = []
	ans4 = []
	assert_equal(c.parse_lin_terms(K4, lin4), ans4)


def test_parse_qua_terms():

	K1 = 3
	qua1 = None
	ans1 = []
	assert_equal(c.parse_qua_terms(K1, qua1), ans1)

	K2 = 2
	qua2 = 'all'
	ans2 = [(0, 0), (0, 1), (1, 1)]
	assert_equal(c.parse_qua_terms(K2, qua2), ans2)

	K3 = 2
	qua3 = [(0, 1)]
	ans3 = [(0, 1)]
	assert_equal(c.parse_qua_terms(K3, qua3), ans3)

	K4 = 2
	qua4 = []
	ans4 = []
	assert_equal(c.parse_qua_terms(K4, qua4), ans4)


def test_split_equal_bins():

	pscore = np.array([0.05, 0.1, 0.2, 0.3, 0.4, 0.5,
	                   0.6, 0.7, 0.8, 0.9, 0.95])
	blocks = 5
	ans = [0, 0.2, 0.4, 0.6, 0.8, 1]

	assert_equal(c.split_equal_bins(pscore, blocks), ans)


def test_sumlessthan():

	g1 = np.array([3, 1, 2, 4, 3, 3])
	sg1 = np.array([1, 2, 3, 3, 3, 4])
	cs11 = np.array([1, 2, 3, 4, 5, 6])
	csg1 = np.array([1, 3, 6, 9, 12, 16])

	ans1 = np.array([5, 1, 2, 6, 5, 5])
	ans2 = np.array([12, 1, 3, 16, 12, 12])
	assert np.array_equal(c.sumlessthan(g1, sg1, cs11), ans1)
	assert np.array_equal(c.sumlessthan(g1, sg1, csg1), ans2)

	g2 = np.array([22, 4, 6, 4, 25, 5])
	sg2 = np.array([4, 4, 5, 6, 22, 25])
	cs12 = np.array([1, 2, 3, 4, 5, 6])
	csg2 = np.array([4, 8, 13, 19, 41, 66])

	ans3 = np.array([5, 2, 4, 2, 6, 3])
	ans4 = np.array([41, 8, 19, 8, 66, 13])
	assert np.array_equal(c.sumlessthan(g2, sg2, cs12), ans3)
	assert np.array_equal(c.sumlessthan(g2, sg2, csg2), ans4)


def test_select_cutoff():

	g1 = np.array([3, 1, 2, 4, 3, 3])
	ans1 = 0
	assert_equal(c.select_cutoff(g1), ans1)

	g2 = np.array([22, 4, 6, 4, 25, 5])
	ans2 = 0.2113248654
	assert np.allclose(c.select_cutoff(g2), ans2)


def test_calc_tstat():

	sample1 = np.array([1, 1, 2, 2, 3, 3, 3, 3, 3, 3,
	                    3, 3, 3, 3, 3, 3, 4, 4, 4, 5])
	sample2 = np.array([5, 5, 5, 5, 5, 5, 5, 4, 4, 4, 4,
	                    4, 4, 4, 4, 4, 4, 3, 3, 3, 2, 2])
	ans = 3.632233

	assert np.allclose(c.calc_tstat(sample1, sample2), ans)


def test_calc_sample_sizes():

	D1 = np.array([0, 1, 0, 1, 0, 1])
	ans1 = (2, 1, 1, 2)
	assert_equal(c.calc_sample_sizes(D1), ans1)

	D2 = np.array([0, 1, 0, 1, 0])
	ans2 = (1, 1, 2, 1)
	assert_equal(c.calc_sample_sizes(D2), ans2)

	D3 = np.array([1, 1, 1, 1, 1, 1])
	ans3 = (0, 3, 0, 3)
	assert_equal(c.calc_sample_sizes(D3), ans3)

	D4 = np.array([0, 0, 0])
	ans4 = (1, 0, 2, 0)
	assert_equal(c.calc_sample_sizes(D4), ans4)


def test_select_blocks():

	pscore1 = np.array([0.05, 0.06, 0.3, 0.4, 0.5, 0.6, 0.7, 0.95, 0.95])
	D1 = np.array([0, 0, 1, 1, 0, 0, 1, 1, 1])
	logodds1 = np.log(pscore1 / (1-pscore1))
	K1 = 1
	ans1 = np.array([0.05, 0.5, 0.5, 0.95])
	test1 = np.array(c.select_blocks(pscore1, logodds1, D1, K1, 0, 1))
	assert np.allclose(test1, ans1)

	pscore2 = np.array([0.05, 0.06, 0.3, 0.4, 0.5, 0.6, 0.7, 0.95, 0.95])
	D2 = np.array([0, 0, 1, 1, 0, 0, 1, 1, 1])
	logodds2 = np.log(pscore1 / (1-pscore1))
	K2 = 2
	ans2 = np.array([0, 1])
	test2 = np.array(c.select_blocks(pscore2, logodds2, D2, K2, 0, 1))
	assert np.allclose(test2, ans2)

