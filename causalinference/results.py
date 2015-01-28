import numpy as np
from scipy.stats import norm

class Results(object):


	def __init__(self, causal):

		self.causal = causal


	def ndiff(self):

		print self.causal.ndiff


	def propensity(self):

		if not hasattr(self.causal, 'pscore'):
			self.causal.propensity()

		print 'Coefficients:', self.causal.pscore['coeff']
		print 'Log-likelihood:', self.causal.pscore['loglike']

	def summary(self):

		header = ('%8s'+'%12s'*4+'%24s') % ('', 'coef', 'std err', 'z', 'P>|z|', '[95% Conf. Int.]')
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
			print ('%8s'+'%12.3f'*6) % (name, coef, se, t, p, lw, up)

