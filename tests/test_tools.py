from nose.tools import *
import numpy as np

import causalinference.utils.tools as t


def test_convert_to_formatting():

	entry_types = ['string', 'float', 'integer', 'float']
	ans = ['s', '.3f', '.0f', '.3f']

	assert_equal(list(t.convert_to_formatting(entry_types)), ans)


def test_add_row():

	entries1 = ('Variable', 'Mean', 'S.d.', 'Mean', 'S.d.', 'Raw diff')
	entry_types1 = ['string']*6
	col_spans1 = [1]*6
	width1 = 80
	ans1 = '       Variable         Mean         S.d.         Mean         S.d.     Raw diff\n'
	assert_equal(t.add_row(entries1, entry_types1, col_spans1, width1), ans1)
	
	entries2 = [12, 13.2, -3.14, 9.8765]
	entry_types2 = ['integer', 'integer', 'float', 'float']
	col_spans2 = [1, 2, 2, 1]
	width2 = 80
	ans2 = '             12                        13                    -3.140        9.877\n'
	assert_equal(t.add_row(entries2, entry_types2, col_spans2, width2), ans2)


def test_add_line():

	width = 30
	ans = '------------------------------\n'

	assert_equal(t.add_line(width), ans)


def test_gen_reg_entries():

	varname = 'Income'
	coef = 0.5
	se = 0.25
	ans1 = 'Income'
	ans2 = 0.5
	ans3 = 0.25
	ans4 = 2
	ans5 = 0.045500
	ans6 = 0.01
	ans7 = 0.99

	v, c, s, z, p, lw, up = t.gen_reg_entries(varname, coef, se)
	assert_equal(v, ans1)
	assert_equal(c, ans2)
	assert_equal(s, ans3)
	assert_equal(z, ans4)
	assert np.allclose(p, ans5)
	assert np.allclose(lw, ans6)
	assert np.allclose(up, ans7)

