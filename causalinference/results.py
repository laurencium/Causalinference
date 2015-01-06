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

