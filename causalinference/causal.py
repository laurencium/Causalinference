import numpy as np
import scipy.linalg

from core import Basic, Stratum, Strata, Propensity, PropensitySelect
from estimators import Estimators, Blocking, Matching, Weighting, OLS
from utils.tools import remove


class CausalModel(Basic):


	def __init__(self, Y, D, X):

		super(CausalModel, self).__init__(Y, D, X)
		self.est = Estimators()
		self._Y_old, self._D_old, self._X_old = Y, D, X


	def restart(self):

		"""
		Reinitializes data to original inputs, and drop any estimated
		results.
		"""

		self.__init__(self._Y_old, self._D_old, self._X_old)
		remove(self, 'pscore')
		remove(self, 'cutoff')
		remove(self, 'blocks')
		remove(self, 'strata')


	def _post_pscore_init(self):

		"""
		Initializes cutoff threshold for trimming and number of
		equal-sized blocks after estimation of propensity score.
		"""

		if not hasattr(self, 'cutoff'):
			self.cutoff = 0.1
		if not hasattr(self, 'blocks'):
			self.blocks = 5


	def propensity(self, lin='all', qua=[]):

		"""
		Estimates via logit the propensity score based on requirements
		on linear terms, and quadratic terms.

		Expected args
		-------------
			lin: string, list
				Column numbers (zero-based) of the original
				covariate matrix X to include linearly. Defaults
				to the string 'all', which uses whole covariate
				matrix.
			qua: list
				Column numbers (zero-based) of the original
				covariate matrix X to include as quadratic
				terms. E.g., [1,3] will include squares of the
				1st and 3rd columns, and the product of these
				two columns.  Default is to not include any
				quadratic terms.
		"""

		self.pscore = Propensity(lin, qua, self)
		self._post_pscore_init()


	def propensity_s(self, lin_B=[], C_lin=1, C_qua=2.71):

		"""
		Estimates via logit the propensity score using the covariate
		selection algorithm of Imbens and Rubin (2015).

		Expected args
		-------------
			lin_B: list
				Column numbers (zero-based) of the original
				covariate matrix X that should be included as
				linear terms regardless. Defaults to empty list,
				meaning every column of X is subjected to the
				selection algorithm.
			C_lin: scalar
				Critical value used in likelihood ratio test
				to decide whether candidate linear terms should
				be included. Defaults to 1 as in Imbens (2014).
			C_qua: scalar
				Critical value used in likelihood ratio test
				to decide whether candidate quadratic terms
				should be included. Defaults to 2.71 as in
				Imbens (2014).

		References
		----------
			Imbens, G. (2014). Matching Methods in Practice: Three
				Examples.
			Imbens, G. & Rubin, D. (2015). Causal Inference in
				Statistics, Social, and Biomedical Sciences: An
				Introduction.
		"""

		self.pscore = PropensitySelect(lin_B, C_lin, C_qua, self)
		self._post_pscore_init()


	def _check_prereq(self, prereq):

		"""
		Checks existence of estimated propensity score or strata.
		"""

		if not hasattr(self, prereq):
			if prereq == 'pscore':
				raise Exception("Missing propensity score")
			if prereq == 'strata':
				raise Exception("Sample has not been stratified")
	

	def trim(self):

		"""
		Trims data based on propensity score to create a subsample with
		better covariate balance. The CausalModel class has cutoff has
		a property, with default value 0.1. User can modify this
		directly, or by calling the method trim_s to have the cutoff
		selected automatically using the algorithm proposed by Crump,
		Hotz, Imbens, and Mitnik (2008).

		Note that trimming the data erases all current estimates.

		References
		----------
			Crump, R., Hotz, V., Imbens, G., & Mitnik, O. (2008).
				Dealing with Limited Overlap in Estimation of
				Average Treatment Effects. Biometrika.
		"""

		self._check_prereq('pscore')
		
		untrimmed = (self.pscore['fitted'] >= self.cutoff) & \
		            (self.pscore['fitted'] <= 1-self.cutoff)
		super(CausalModel, self).__init__(self.Y[untrimmed],
		                                  self.D[untrimmed],
						  self.X[untrimmed])
		self.pscore['fitted'] = self.pscore['fitted'][untrimmed]
		self.est = Estimators()


	def _select_cutoff(self):

		"""
		Selects cutoff value for propensity score used in trimming
		function using algorithm suggested by Crump, Hotz, Imbens,
		and Mitnik (2008).
		
		References
		----------
			Crump, R., Hotz, V., Imbens, G., & Mitnik, O. (2008).
				Dealing with Limited Overlap in Estimation of
				Average Treatment Effects. Biometrika.
		"""

		g = 1 / (self.pscore['fitted'] * (1-self.pscore['fitted']))
		order = np.argsort(g)
		h = g[order].cumsum()/np.square(xrange(1,self.N+1))
		
		self.cutoff = 0.5 - np.sqrt(0.25-1/g[order[h.argmin()]])

	
	def trim_s(self):

		"""
		Trims data based on propensity score using cutoff selected using
		algorithm suggested by Crump, Hotz, Imbens, and Mitnik (2008).
		Algorithm is of order O(N).

		References
		----------
			Crump, R., Hotz, V., Imbens, G., & Mitnik, O. (2008).
				Dealing with Limited Overlap in Estimation of
				Average Treatment Effects. Biometrika.
		"""

		self._check_prereq('pscore')

		self._select_cutoff()
		self.trim()


	def stratify(self):

		"""
		Stratifies the sample based on propensity score. If the class
		attribute blocks is a number, then equal-sized bins will be
		created. Otherwise if blocks is a list of bin boundaries then
		the bins will be created accordingly.
		"""

		self._check_prereq('pscore')
		remove(self, 'strata')

		phat = self.pscore['fitted']
		if isinstance(self.blocks, (int, long)):
			q = np.linspace(0, 100, self.blocks+1)[1:-1]
			self.blocks = [0] + list(np.percentile(phat, q)) + [1]

		self.blocks.sort()
		self.blocks[0] *= 0.99  # adjust to not drop obs w/ min pscore
		bins = [None] * (len(self.blocks)-1)
		for i in xrange(len(self.blocks)-1):
			subclass = (phat>self.blocks[i]) & \
			           (phat<=self.blocks[i+1])
			Y = self.Y[subclass]
			D = self.D[subclass]
			X = self.X[subclass]
			bins[i] = Stratum(Y, D, X, phat[subclass])

		self.strata = Strata(bins)


	def _select_blocks(self, e, l, e_min, e_max):

		"""
		Selects propensity bins recursively for blocking estimator using
		algorithm suggested by Imbens and Rubin (2015).

		Expected args
		-------------
			e: array-like
				Vector of estimated propensity scores for the
				whole sample.
			l: array-like
				Vector of log odds ratio for the whole sample.
			e_min: scalar
				Lower boundary of current propensity bin.
			e_max: scalar
				Upper boundary of current propensity bin.

		Returns
		-------
			List containing bin boundaries.

		References
		----------
			Imbens, G. & Rubin, D. (2015). Causal Inference in
				Statistics, Social, and Biomedical Sciences:
				An Introduction.
		"""

		scope = (e >= e_min) & (e <= e_max)
		t, c = (scope & (self.D==1)), (scope & (self.D==0))

		# unravel if no significant difference in log odds ratio
		t_stat = (l[t].mean()-l[c].mean()) / \
		         np.sqrt(l[t].var()/t.sum() + l[c].var()/c.sum())
		if t_stat <= 1.96:
			return [e_min, e_max]

		mid = e[e <= np.median(e[scope])].max()
		left = (e <= mid) & scope
		right = (e > mid) & scope
		N_left = left.sum()
		N_right = right.sum()
		N_left_t = (left & (self.D==1)).sum()
		N_right_t = (right & (self.D==1)).sum()

		# unravel if sample sizes are too small
		if np.min([N_left, N_right]) <= self.K+2:
			return [e_min, e_max]
		if np.min([N_left_t, N_left-N_left_t, N_right_t,
		           N_right-N_right_t]) <= 3:
			return [e_min, e_max]

		# split bin and recurse on left and right
		return self._select_blocks(e, l, e[left].min(), mid) + \
		       self._select_blocks(e, l, mid, e[right].max())


	def stratify_s(self):

		"""
		Stratifies the sample based on propensity score using bin
		selection algorithm suggested by Imbens and Rubin (2015).
		Algorithm is of order O(N log N).

		References
		----------
			Imbens, G. & Rubin, D. (2015). Causal Inference in
				Statistics, Social, and Biomedical Sciences:
				An Introduction.
		"""

		self._check_prereq('pscore')

		phat = self.pscore['fitted']
		l = np.log(phat / (1+phat))  # log odds ratio
		e_min = phat.min()
		e_max = phat.max()
		self.blocks = list(set(self._select_blocks(phat, l,
		                                           e_min, e_max)))
		self.stratify()


	def blocking(self):

		"""
		Computes average treatment effects using regression within
		blocks. Sample must be stratified first.
		"""

		self._check_prereq('strata')
		self.est['blocking'] = Blocking(self)


	def matching(self, wmat='inv', m=1, xbias=False):

		"""
		Estimates average treatment effects using matching with
		replacement.

		By default, the weighting matrix used in measuring distance is
		the inverse variance matrix. The Mahalanobis metric or other
		arbitrary weighting matrices can also be used instead.

		The number of matches per subject can also be specified. Tied
		entries are included, so the number of matches can be greater
		than specified for some subjects.

		Bias correction can optionally be done. For treated units, the
		bias resulting from imperfect matches is estimated by
			(X_t - X_c[matched]) * b,
		where b is the estimated coefficient from regressiong
		Y_c[matched] on X_c[matched]. For control units, the analogous
		procedure is used. For details, see Imbens and Rubin (2015).

		Expected args
		-------------
			wmat: string or matrix, ndarray
				Weighting matrix to be used in norm calcuation.
				Acceptable values are 'inv' for inverse
				variance weighting, 'maha' for Mahalanobis
				metric, or any arbitrary K-by-K matrix.
			m: integer
				The number of units to match to a given
				subject. Defaults to 1.
			xbias: Boolean
				Correct bias resulting from imperfect matches
				or not; defaults to no correction.

		References
		----------
			Imbens, G. & Rubin, D. (2015). Causal Inference in
				Statistics, Social, and Biomedical Sciences:
				An Introduction.
		"""

		self.est['matching'] = Matching(wmat, m, xbias, self)


	def weighting(self):

		"""
		Computes treatment effects using the Horvitz-Thompson weighting
		estimator modified to incorporate covariates. Estimator
		possesses the so-called 'double robustness' property. See
		Lunceford and Davidian (2004) for details.

		References
		----------
			Lunceford, J. K., & Davidian, M. (2004). Stratification
				and weighting via the propensity score in
				estimation of causal treatment effects: a
				comparative study. Statistics in Medicine.
		"""

		self._check_prereq('pscore')
		self.est['weighting'] = Weighting(self)


	def ols(self):

		"""
		Estimates average treatment effects using least squares.

		The OLS estimate of ATT can be shown to be equal to
			mean(Y_t) - (alpha + beta * mean(X_t)),
		where alpha and beta are coefficients from the control group
		regression:
			Y_c = alpha + beta * X_c + e.
		ATC can be estimated analogously. Subsequently, ATE can be
		estimated as sample weighted average of the ATT and ATC
		estimates.
		"""

		self.est['ols'] = OLS(self)

