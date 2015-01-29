import numpy as np


class Covariates(object):

	"""
	Contains summary statistics for the covariate variables.

	One of the summary statistics is the normalized differenced, defined as
	the difference in group means, scaled by the square root of the average
	of the two within-group variances. Large values indicate that simple
	linear adjustment methods may not be adequate for removing biases that
	are associated with differences in covariates.

	Unlike t-statistic, normalized differences do not, in expectation,
	increase with sample size, and thus is more appropriate for assessing
	balance.
	"""

	def __init__(self, X_c, X_t):

		self._dict = {}
		self._dict['mean_c'] = X_c.mean(0)
		self._dict['mean_t'] = X_t.mean(0)
		self._dict['sd_c'] = np.sqrt(X_c.var(0))
		self._dict['sd_t'] = np.sqrt(X_t.var(0))
		self._dict['ndiff'] = (self._dict['mean_t'] - \
		                       self._dict['mean_c']) / \
		                      np.sqrt((self._dict['sd_t']**2 + \
				               self._dict['sd_c']**2)/2)


	def __dir__(self):

		return ['mean_c', 'mean_t', 'sd_c', 'sd_t', 'ndiff']


	def __getitem__(self, key):

		return self._dict[key]


	def __str__(self):

		return "Placeholder string."

