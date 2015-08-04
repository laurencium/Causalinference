from nose.tools import *
import numpy as np

from causalinference.utils.tools import add_row

def test_add_row():

	entries = ('Variable', 'Mean', 'S.d.', 'Mean', 'S.d.', 'Raw diff')
	entry_types = ['string']*6
	col_spans = [1]*6
	width = 80
	
	ans = '       Variable         Mean         S.d.         Mean         S.d.     Raw diff\n'
	assert_equal(add_row(entries, entry_types, col_spans, width), ans)

