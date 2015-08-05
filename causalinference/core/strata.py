import numpy as np
import scipy.linalg
from itertools import izip

from ..utils.tools import Printer


class Strata(object):

	"""
	List-like object containing the stratified propensity bins.
	"""

	def __init__(self, strata, subsets, pscore):

		self._strata = strata
		for stratum, subset in izip(self._strata, subsets):
			pscore_sub = pscore[subset]
			stratum.raw_data._dict['pscore'] = pscore_sub
			stratum.summary_stats._summarize_pscore(pscore_sub)
			

	def __len__(self):

		return len(self._strata)


	def __getitem__(self, index):

		return self._strata[index]


	def __str__(self):

		p = Printer()

		output = '\n'
		output += 'Stratification Summary\n\n'

		entries = ('', 'Propensity score', '', 'Ave. p-score', 'Within')
		span = [1, 2, 2, 2, 1]
		etype = ['string']*5
		output += p.write_row(entries, span, etype)

		entries = ('Stratum', 'Min.', 'Max.', 'N_c', 'N_t',
		           'Controls', 'Treated', 'Est.')
		span = [1]*8
		etype = ['string']*8
		output += p.write_row(entries, span, etype)
		output += p.write_row('-'*p.table_width, [1], ['string'])

		strata = self._strata
		etype = ['integer', 'float', 'float', 'integer', 'integer',
		         'float', 'float', 'float']
		for i in xrange(len(strata)):

			c, t = strata[i].controls, strata[i].treated
			entries = (i+1, strata[i].pscore['min'],
			           strata[i].pscore['max'], strata[i].N_c,
				   strata[i].N_t,
				   strata[i].pscore['fitted'][c].mean(),
				   strata[i].pscore['fitted'][t].mean(),
				   strata[i].within)
			output += p.write_row(entries, span, etype)

		return output

