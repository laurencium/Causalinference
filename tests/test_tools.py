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

