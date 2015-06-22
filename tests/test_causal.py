from nose.tools import *
import numpy as np

from causalinference import CausalModel
from tools import random_data


def test_est_propensity():

	D = np.array([0, 0, 1, 1])
	X = np.array([[1, 2], [9, 7], [1, 4], [9, 6]])
	Y = random_data(D_cur=D, X_cur=X)
	causal = CausalModel(Y, D, X)

	causal.est_propensity()
	lin = [0, 1]
	qua = []
	coef = np.array([-2.1505403, -0.3671654, 0.8392352])
	loglike = -2.567814
	fitted = np.array([0.3016959, 0.6033917, 0.6983041, 0.3966083])
	se = np.array([3.8953529, 0.6507885, 1.3595614])
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
	causal = CausalModel(Y, D, X)

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


def test_parse_lin_terms():

	K1 = 4
	lin1 = None
	ans1 = []
	assert_equal(CausalModel._parse_lin_terms(K1, lin1), ans1)

	K2 = 2
	lin2 = 'all'
	ans2 = [0, 1]
	assert_equal(CausalModel._parse_lin_terms(K2, lin2), ans2)

	K3 = 2
	lin3 = [1]
	ans3 = [1]
	assert_equal(CausalModel._parse_lin_terms(K3, lin3), ans3)

	K4 = 2
	lin4 = []
	ans4 = []
	assert_equal(CausalModel._parse_lin_terms(K4, lin4), ans4)


def test_parse_qua_terms():

	K1 = 3
	qua1 = None
	ans1 = []
	assert_equal(CausalModel._parse_qua_terms(K1, qua1), ans1)

	K2 = 2
	qua2 = 'all'
	ans2 = [(0, 0), (0, 1), (1, 1)]
	assert_equal(CausalModel._parse_qua_terms(K2, qua2), ans2)

	K3 = 2
	qua3 = [(0, 1)]
	ans3 = [(0, 1)]
	assert_equal(CausalModel._parse_qua_terms(K3, qua3), ans3)

	K4 = 2
	qua4 = []
	ans4 = []
	assert_equal(CausalModel._parse_qua_terms(K4, qua4), ans4)


def test_sumlessthan():

	g1 = np.array([3, 1, 2, 4, 3, 3])
	sg1 = np.array([1, 2, 3, 3, 3, 4])
	cs11 = np.array([1, 2, 3, 4, 5, 6])
	csg1 = np.array([1, 3, 6, 9, 12, 16])

	ans1 = np.array([5, 1, 2, 6, 5, 5])
	ans2 = np.array([12, 1, 3, 16, 12, 12])
	assert np.array_equal(CausalModel._sumlessthan(g1, sg1, cs11), ans1)
	assert np.array_equal(CausalModel._sumlessthan(g1, sg1, csg1), ans2)

	g2 = np.array([22, 4, 6, 4, 25, 5])
	sg2 = np.array([4, 4, 5, 6, 22, 25])
	cs12 = np.array([1, 2, 3, 4, 5, 6])
	csg2 = np.array([4, 8, 13, 19, 41, 66])

	ans3 = np.array([5, 2, 4, 2, 6, 3])
	ans4 = np.array([41, 8, 19, 8, 66, 13])
	assert np.array_equal(CausalModel._sumlessthan(g2, sg2, cs12), ans3)
	assert np.array_equal(CausalModel._sumlessthan(g2, sg2, csg2), ans4)


def test_select_cutoff():

	g1 = np.array([3, 1, 2, 4, 3, 3])
	ans1 = 0
	assert_equal(CausalModel._select_cutoff(g1), ans1)

	g2 = np.array([22, 4, 6, 4, 25, 5])
	ans2 = 0.2113248654
	assert np.allclose(CausalModel._select_cutoff(g2), ans2)

