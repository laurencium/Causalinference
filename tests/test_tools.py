from nose.tools import *
import numpy as np

from causalinference.utils.tools import add_row

def test_add_row():

	entries1 = ('Variable', 'Mean', 'S.d.', 'Mean', 'S.d.', 'Raw diff')
	entry_types1 = ['string']*6
	col_spans1 = [1]*6
	width1 = 80
	ans1 = '       Variable         Mean         S.d.         Mean         S.d.     Raw diff\n'
	assert_equal(add_row(entries1, entry_types1, col_spans1, width1), ans1)
	
	entries2 = [12, 13.2, -3.14, 9.8765]
	entry_types2 = ['integer', 'integer', 'float', 'float']
	col_spans2 = [1, 2, 2, 1]
	width2 = 80
	ans2 = '             12                        13                    -3.140        9.877\n'
	assert_equal(add_row(entries2, entry_types2, col_spans2, width2), ans2)

