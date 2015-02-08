import numpy as np
import scipy.linalg

from basic import Basic
from ..utils.tools import cache_readonly, Printer


class Stratum(Basic):

	"""
	Class containing as attributes propensity-bin-specific data, including
	sample sizes, covariate summaries, propensity-score-related data, and
	within-stratum treatment effect estimates and standard errors.
	"""

	def __init__(self, Y, D, X, pscore):

		super(Stratum, self).__init__(Y, D, X)
		self.pscore = {'fitted': pscore, 'min': pscore.min(),
		               'mean': pscore.mean(), 'max': pscore.max()}


	@cache_readonly
	def within(self):
		return self._compute_within()


	@cache_readonly
	def se(self):
		return self._compute_se()


	def _compute_within(self):
	
		"""
		Computes within-block treatment effect estimate by regressing
		the within-block observed outcomes on covarites, treatment
		indicator, and a constant term. The within-block estimate is
		then the estimated coefficient for the treatment indicator.
		"""

		self._Z = np.empty((self.N, self.K+2))  # create design matrix
		self._Z[:,0], self._Z[:,1], self._Z[:,2:] = 1, self.D, self.X
		Q, self._R = np.linalg.qr(self._Z)  # save R for later use
		self._olscoef = scipy.linalg.solve_triangular(self._R,
		                                               Q.T.dot(self.Y))

		return self._olscoef[1]


	def _compute_se(self):

		"""
		Computes standard error for within-block treatment effect
		estimate.
		
		If Z denotes the design matrix (i.e., covariates, treatment
		indicator, and a column of ones) and u denotes the vector of
		least squares residual, then the variance estimator can be
		found by computing White's heteroskedasticity robust covariance
		matrix:
			inv(Z'Z) Z'diag(u^2)Z inv(Z'Z).
		The diagonal entry corresponding to the treatment indicator of
		this matrix is the appropriate variance estimate for the block.
		"""

		if not hasattr(self, '_olscoef'):
			self._within = self._compute_within()
		u = self.Y - self._Z.dot(self._olscoef)
		A = np.linalg.inv(np.dot(self._R.T, self._R))
		B = np.dot(u[:,None]*self._Z, A[:,1])

		return np.sqrt(np.dot(B.T, B))


class Strata(object):

	"""
	List-like object containing the list of stratified propensity bins.
	"""

	def __init__(self, strata):

		self._strata = strata


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

