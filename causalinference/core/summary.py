from __future__ import division
import numpy as np

from data import Dict
from ..utils.tools import Printer


class Summary(Dict):

	"""
	Dictionary-like class containing summary statistics for input data.

	One of the summary statistics is the normalized difference between
	covariates. Large values indicate that simple linear adjustment methods
	may not be adequate for removing biases that are associated with
	differences in covariates.
	"""

	def __init__(self, data):

		self._dict = dict()

		self._dict['N'] = data['N']
		self._dict['N_c'], self._dict['N_t'] = data['N_c'], data['N_t']
		self._dict['Y_c_mean'] = data['Y_c'].mean()
		self._dict['Y_t_mean'] = data['Y_t'].mean()
		self._dict['Y_c_sd'] = np.sqrt(data['Y_c'].var(ddof=1))
		self._dict['Y_t_sd'] = np.sqrt(data['Y_t'].var(ddof=1))
		self._dict['rdiff'] = self['Y_t_mean'] - self['Y_c_mean']
		self._dict['X_c_mean'] = data['X_c'].mean(0)
		self._dict['X_t_mean'] = data['X_t'].mean(0)
		self._dict['X_c_sd'] = np.sqrt(data['X_c'].var(0, ddof=1))
		self._dict['X_t_sd'] = np.sqrt(data['X_t'].var(0, ddof=1))
		self._dict['ndiff'] = calc_ndiff(self['X_c_mean'],
		                                 self['X_t_mean'],
						 self['X_c_sd'],
						 self['X_t_sd'])


	def _summarize_pscore(self, pscore):

		self._dict['p_min'] = pscore.min()
		self._dict['p_max'] = pscore.max()
		self._dict['p_mean'] = pscore.mean()


	def __str__(self):

		p = Printer()
		N_c, N_t = self['N_c'], self['N_t']
		K = self['X_c_mean'].shape[0]
		mean_c, mean_t = self['X_c_mean'], self['X_t_mean']
		sd_c, sd_t = self['X_c_sd'], self['X_t_mean']
		ndiff = self['ndiff']
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
			

def calc_ndiff(mean_c, mean_t, sd_c, sd_t):

	return (mean_t-mean_c) / np.sqrt((sd_c**2+sd_t**2)/2)

