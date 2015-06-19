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

