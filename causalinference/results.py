import numpy as np
from scipy.stats import norm

class Results(object):


	def __init__(self, causal):

		self.causal = causal
		self.table_width = 80


	def _varnames(self, varnums):

		return ['X'+str(varnum+1) for varnum in varnums]


	def _make_row(self, entries):

		col_width = self.table_width // len(entries)
		first_col_width = col_width + self.table_width % len(entries)

		return ('%'+str(first_col_width)+'s' + ('%'+str(col_width)+'.3f')*(len(entries)-1)) % entries


	def ndiff(self):

		varnames = self._varnames(xrange(self.causal.K))
		X_t_mean = self.causal.X_t.mean(0)
		X_t_sd = np.sqrt(self.causal.X_t.var(0))
		X_c_mean = self.causal.X_c.mean(0)
		X_c_sd = np.sqrt(self.causal.X_c.var(0))

		for i in xrange(self.causal.K):
			print self._make_row((varnames[i], X_t_mean[i], X_t_sd[i], X_c_mean[i], X_c_sd[i], self.causal.ndiff[i]))


	def propensity(self):

		if not hasattr(self.causal, 'pscore'):
			self.causal.propensity()

		print 'Coefficients:', self.causal.pscore['coeff']
		print 'Log-likelihood:', self.causal.pscore['loglike']

	def summary(self):

		header = ('%8s'+'%12s'*4+'%24s') % ('', 'est', 'std err', 'z', 'P>|z|', '[95% Conf. Int.]')
		print header
		print '-' * len(header)
		tuples = (('ATE', self.causal.ate, self.causal.ate_se),
		         ('ATT', self.causal.att, self.causal.att_se),
			 ('ATC', self.causal.atc, self.causal.atc_se))
		for (name, coef, se) in tuples:
			t = coef / se
			p = 1 - norm.cdf(np.abs(t))
			lw = coef - 1.96*se
			up = coef + 1.96*se
			print self._make_row((name, coef, se, t, p, lw, up))

