from nose.tools import *
import numpy as np

from causalinference import CausalModel


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

