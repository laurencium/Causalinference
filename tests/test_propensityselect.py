from nose.tools import *
import numpy as np

import causalinference.core.data as d
import causalinference.core.propensity as p
from utils import random_data


def test_get_excluded_lin():

	K1 = 4
	included1 = []
	ans1 = [0, 1, 2, 3]
	assert_equal(p.get_excluded_lin(K1, included1), ans1)

	K2 = 4
	included2 = [3, 1]
	ans2 = [0, 2]
	assert_equal(p.get_excluded_lin(K2, included2), ans2)

	K3 = 3
	included3 = [0, 1, 2]
	ans3 = []
	assert_equal(p.get_excluded_lin(K3, included3), ans3)


def test_get_excluded_qua():

	lin1 = [0, 2, 3]
	qua1 = [(0, 3), (3, 3)]
	ans1 = [(0, 0), (0, 2), (2, 2), (2, 3)]
	assert_equal(p.get_excluded_qua(lin1, qua1), ans1)

	lin2 = [1, 2]
	qua2 = []
	ans2 = [(1, 1), (1, 2), (2, 2)]
	assert_equal(p.get_excluded_qua(lin2, qua2), ans2)

	lin3 = [8, 5]
	qua3 = [(8, 8), (8, 5), (5, 5)]
	ans3 = []
	assert_equal(p.get_excluded_qua(lin3, qua3), ans3)


def test_calc_loglike():

	X_c = np.array([[1, 2], [3, 7]])
	X_t = np.array([[1, 4], [3, 6]])
	lin = [1]
	qua = [(0, 0)]
	ans = -2.567814
	assert np.allclose(p.calc_loglike(X_c, X_t, lin, qua), ans)


def test_select_lin():

	Y, D, X = random_data(K=4)
	X_c_random, X_t_random = X[D==0], X[D==1]

	lin1 = [0, 1, 2, 3]
	C1 = np.random.rand(1)
	ans1 = [0, 1, 2, 3]
	assert_equal(p.select_lin(X_c_random, X_t_random, lin1, C1), ans1)

	X_c = np.array([[1, 2], [9, 7]])
	X_t = np.array([[1, 4], [9, 6]])

	lin2 = []
	C2 = 0.07
	ans2 = []
	assert_equal(p.select_lin(X_c, X_t, lin2, C2), ans2)

	lin3 = []
	C3 = 0.06
	ans3 = [1, 0]
	assert_equal(p.select_lin(X_c, X_t, lin3, C3), ans3)

	lin4 = [1]
	C4 = 0.35
	ans4 = [1]
	assert_equal(p.select_lin(X_c, X_t, lin4, C4), ans4)

	lin5 = [1]
	C5 = 0.34
	ans5 = [1, 0]
	assert_equal(p.select_lin(X_c, X_t, lin5, C5), ans5)


def test_select_lin_terms():

	Y, D, X = random_data(K=4)
	X_c_random, X_t_random = X[D==0], X[D==1]

	lin1 = [3, 0, 1]
	C1 = np.inf
	ans1 = [3, 0, 1]
	assert_equal(p.select_lin_terms(X_c_random, X_t_random, lin1, C1), ans1)

	lin2 = [2]
	C2 = 0
	ans2 = [2, 0, 1, 3]
	assert_equal(p.select_lin_terms(X_c_random, X_t_random, lin2, C2), ans2)
	
	lin3 = []
	C3 = 0
	ans3 = [0, 1, 2, 3]
	assert_equal(p.select_lin_terms(X_c_random, X_t_random, lin3, C3), ans3)
	
	lin4 = [3, 1]
	C4 = -34.234
	ans4 = [3, 1, 0, 2]
	assert_equal(p.select_lin_terms(X_c_random, X_t_random, lin4, C4), ans4)

	X_c = np.array([[1, 2], [9, 7]])
	X_t = np.array([[1, 4], [9, 7]])

	lin5 = []
	C5 = 0.06
	ans5 = [1, 0]
	assert_equal(p.select_lin_terms(X_c, X_t, lin5, C5), ans5)


def test_select_qua():

	Y, D, X = random_data()
	X_c_random, X_t_random = X[D==0], X[D==1]

	lin1 = [1, 0]
	qua1 = [(1, 0), (0, 0), (1, 1)]
	C1 = np.random.rand(1)
	ans1 = [(1, 0), (0, 0), (1, 1)]
	assert_equal(p.select_qua(X_c_random, X_t_random, lin1, qua1, C1), ans1)

	lin2 = [1]
	qua2 = [(1, 1)]
	C2 = np.random.rand(1)
	ans2 = [(1, 1)]
	assert_equal(p.select_qua(X_c_random, X_t_random, lin2, qua2, C2), ans2)

	X_c = np.array([[7, 8], [3, 10], [7, 10]])
	X_t = np.array([[4, 7], [5, 10], [9, 8]])

	lin3 = [0, 1]
	qua3 = []
	C3 = 1.2
	ans3 = []
	assert_equal(p.select_qua(X_c, X_t, lin3, qua3, C3), ans3)

	lin4 = [0, 1]
	qua4 = []
	C4 = 1.1
	ans4 = [(1, 1), (0, 1), (0, 0)]
	assert_equal(p.select_qua(X_c, X_t, lin4, qua4, C4), ans4)

	lin5 = [0, 1]
	qua5 = [(1, 1)]
	C5 = 2.4
	ans5 = [(1, 1)]
	assert_equal(p.select_qua(X_c, X_t, lin5, qua5, C5), ans5)

	lin6 = [0, 1]
	qua6 = [(1, 1)]
	C6 = 2.3
	ans6 = [(1, 1), (0, 1), (0, 0)]
	assert_equal(p.select_qua(X_c, X_t, lin6, qua6, C6), ans6)

	lin7 = [0, 1]
	qua7 = [(1, 1), (0, 1)]
	C7 = 3.9
	ans7 = [(1, 1), (0, 1)]
	assert_equal(p.select_qua(X_c, X_t, lin7, qua7, C7), ans7)

	lin8 = [0, 1]
	qua8 = [(1, 1), (0, 1)]
	C8 = 3.8
	ans8 = [(1, 1), (0, 1), (0, 0)]
	assert_equal(p.select_qua(X_c, X_t, lin8, qua8, C8), ans8)


def test_select_qua_terms():

	Y, D, X = random_data()
	X_c_random, X_t_random = X[D==0], X[D==1]

	lin1 = [0, 1]
	C1 = np.inf
	ans1 = []
	assert_equal(p.select_qua_terms(X_c_random, X_t_random, lin1, C1), ans1)

	lin2 = [1, 0]
	C2 = 0
	ans2 = [(1, 1), (1, 0), (0, 0)]
	assert_equal(p.select_qua_terms(X_c_random, X_t_random, lin2, C2), ans2)
	
	lin3 = [0]
	C3 = -983.340
	ans3 = [(0, 0)]
	assert_equal(p.select_qua_terms(X_c_random, X_t_random, lin3, C3), ans3)
	
	lin4 = []
	C4 = 34.234
	ans4 = []
	assert_equal(p.select_qua_terms(X_c_random, X_t_random, lin4, C4), ans4)

	X_c = np.array([[7, 8], [3, 10], [7, 10]])
	X_t = np.array([[4, 7], [5, 10], [9, 8]])

	lin5 = [0, 1]
	C5 = 1.1
	ans5 = [(1, 1), (0, 1), (0, 0)]
	assert_equal(p.select_qua_terms(X_c, X_t, lin5, C5), ans5)


def test_propensityselect():

	D = np.array([0, 0, 0, 1, 1, 1])
	X = np.array([[7, 8], [3, 10], [7, 10], [4, 7], [5, 10], [9, 8]])
	Y = random_data(D_cur=D, X_cur=X)
	data = d.Data(Y, D, X)

	propensity1 = p.PropensitySelect(data, [], 1, 2.71)
	lin1 = [1]
	qua1 = []
	coef1 = np.array([6.5424027, -0.7392041])
	loglike1 = -3.627939
	fitted1 = np.array([0.6522105, 0.2995088, 0.2995088,
	                   0.7970526, 0.2995088, 0.6522105])
	se1 = np.array([6.8455179, 0.7641445])
	keys = {'lin', 'qua', 'coef', 'loglike', 'fitted', 'se'}
	
	assert_equal(propensity1['lin'], lin1)
	assert_equal(propensity1['qua'], qua1)
	assert np.allclose(propensity1['coef'], coef1)
	assert np.allclose(propensity1['loglike'], loglike1)
	assert np.allclose(propensity1['fitted'], fitted1)
	assert np.allclose(propensity1['se'], se1)
	assert_equal(set(propensity1.keys()), keys)


	propensity2 = p.PropensitySelect(data, [0, 1], 1, 2.71)
	lin2 = [0, 1]
	qua2 = []
	coef2 = np.array([6.8066090, -0.0244874, -0.7524939])
	loglike2 = -3.626517
	fitted2 = np.array([0.6491366, 0.3117840, 0.2911631,
	                    0.8086407, 0.3013733, 0.6379023])
	se2 = np.array([8.5373779, 0.4595191, 0.8106499])

	assert_equal(propensity2['lin'], lin2)
	assert_equal(propensity2['qua'], qua2)
	assert np.allclose(propensity2['coef'], coef2)
	assert np.allclose(propensity2['loglike'], loglike2)
	assert np.allclose(propensity2['fitted'], fitted2)
	assert np.allclose(propensity2['se'], se2)

