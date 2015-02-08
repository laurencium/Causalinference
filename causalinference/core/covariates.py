import numpy as np

from ..utils.tools import Printer


class Covariates(object):

	"""
	Dictionary-like class containing summary statistics for the covariate
	variables.

	One of the summary statistics is the normalized differenced, defined as
	the difference in group means, scaled by the square root of the average
	of the two within-group variances. Large values indicate that simple
	linear adjustment methods may not be adequate for removing biases that
	are associated with differences in covariates.

	Unlike t-statistic, normalized differences do not, in expectation,
	increase with sample size, and thus is more appropriate for assessing
	balance.
	"""

	def __init__(self, model):

		self._model = model
		X_c, X_t = self._model.X_c, self._model.X_t
		self._dict = {'mean_c': X_c.mean(0), 'mean_t': X_t.mean(0)}
		self._dict['sd_c'] = np.sqrt(X_c.var(0))
		self._dict['sd_t'] = np.sqrt(X_t.var(0))
		mean_diff = self._dict['mean_t'] - self._dict['mean_c']
		var_sum = self._dict['sd_c']**2 + self._dict['sd_t']**2
		self._dict['ndiff'] = mean_diff / np.sqrt(var_sum/2)


	def __getitem__(self, key):

		return self._dict[key]


	def __str__(self):

		p = Printer()
		N_c, N_t = self._model.N_c, self._model.N_t
		K = self._model.K
		mean_c, mean_t = self._dict['mean_c'], self._dict['mean_t']
		sd_c, sd_t = self._dict['sd_c'], self._dict['sd_t']
		ndiff = self._dict['ndiff']
		varnames = ['X'+str(i) for i in xrange(K)]
		
		output = '\n'
		output += 'Covariates Summary\n\n'

		entries = ('', 'Controls (N_c='+str(N_c)+')',
		           'Treated (N_t='+str(N_t)+')', '')
		span = [1, 2, 2, 1]
		etype = ['string']*4
		output += p.write_row(entries, span, etype)

		entries = ('Covariate', 'Mean', 'S.d.', 'Mean', 'S.d.', 'Nor-diff')
		span = [1]*6
		etype = ['string']*6
		output += p.write_row(entries, span, etype)
		output += p.write_row('-'*p.table_width, [1], ['string'])
		
		etype = ['string'] + ['float']*5
		for i in xrange(K):
			entries = (varnames[i], mean_c[i], sd_c[i], mean_t[i],
			           sd_t[i], ndiff[i])
			output += p.write_row(entries, span, etype)

		return output
			

	def keys(self):

		return self._dict.keys()

