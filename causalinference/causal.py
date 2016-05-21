from __future__ import division
import numpy as np
from itertools import combinations_with_replacement

from .core import Data, Summary, Propensity, PropensitySelect, Strata
from .estimators import OLS, Blocking, Weighting, Matching, Estimators


class CausalModel(object):

	"""
	Class that provides the main tools of Causal Inference.
	"""

	def __init__(self, Y, D, X):

		self.old_data = Data(Y, D, X)
		self.reset()


	def reset(self):

		"""
		Reinitializes data to original inputs, and drops any estimated
		results.
		"""

		Y, D, X = self.old_data['Y'], self.old_data['D'], self.old_data['X']
		self.raw_data = Data(Y, D, X)
		self.summary_stats = Summary(self.raw_data)
		self.propensity = None
		self.cutoff = None
		self.blocks = None
		self.strata = None
		self.estimates = Estimators()


	def est_propensity(self, lin='all', qua=None):

		"""
		Estimates the propensity scores given list of covariates to
		include linearly or quadratically.

		The propensity score is the conditional probability of
		receiving the treatment given the observed covariates.
		Estimation is done via a logistic regression.

		Parameters
		----------
		lin: string or list, optional
			Column numbers (zero-based) of variables of
			the original covariate matrix X to include
			linearly. Defaults to the string 'all', which
			uses whole covariate matrix.
		qua: list, optional
			Tuples indicating which columns of the original
			covariate matrix to multiply and include. E.g.,
			[(1,1), (2,3)] indicates squaring the 2nd column
			and including the product of the 3rd and 4th
			columns. Default is to not include any
			quadratic terms.
		"""

		lin_terms = parse_lin_terms(self.raw_data['K'], lin)
		qua_terms = parse_qua_terms(self.raw_data['K'], qua)

		self.propensity = Propensity(self.raw_data, lin_terms, qua_terms)
		self.raw_data._dict['pscore'] = self.propensity['fitted']
		self._post_pscore_init()


	def est_propensity_s(self, lin_B=None, C_lin=1, C_qua=2.71):
	
		"""
		Estimates the propensity score with covariates selected using
		the algorithm suggested by [1]_.

		The propensity score is the conditional probability of
		receiving the treatment given the observed covariates.
		Estimation is done via a logistic regression.

		The covariate selection algorithm is based on a sequence
		of likelihood ratio tests.

		Parameters
		----------
		lin_B: list, optional
			Column numbers (zero-based) of variables of
			the original covariate matrix X to include
			linearly. Defaults to empty list, meaning
			every column of X is subjected to the
			selection algorithm.
		C_lin: scalar, optional
			Critical value used in likelihood ratio tests
			to decide whether candidate linear terms should
			be included. Defaults to 1 as in [1]_.
		C_qua: scalar, optional
			Critical value used in likelihood ratio tests
			to decide whether candidate quadratic terms
			should be included. Defaults to 2.71 as in
			[1]_.

		References
		----------
		.. [1] Imbens, G. & Rubin, D. (2015). Causal Inference in
			Statistics, Social, and Biomedical Sciences: An
			Introduction.
		"""

		lin_basic = parse_lin_terms(self.raw_data['K'], lin_B)

		self.propensity = PropensitySelect(self.raw_data, lin_basic,
		                                   C_lin, C_qua)
		self.raw_data._dict['pscore'] = self.propensity['fitted']
		self._post_pscore_init()


	def trim(self):

		"""
		Trims data based on propensity score to create a subsample with
		better covariate balance.
		
		The default cutoff value is set to 0.1. To set a custom cutoff
		value, modify the object attribute named cutoff directly.

		This method should only be executed after the propensity score
		has been estimated.
		"""

		if 0 < self.cutoff <= 0.5:
			pscore = self.raw_data['pscore']
			keep = (pscore >= self.cutoff) & (pscore <= 1-self.cutoff)
			Y_trimmed = self.raw_data['Y'][keep]
			D_trimmed = self.raw_data['D'][keep]
			X_trimmed = self.raw_data['X'][keep]
			self.raw_data = Data(Y_trimmed, D_trimmed, X_trimmed)
			self.raw_data._dict['pscore'] = pscore[keep]
			self.summary_stats = Summary(self.raw_data)
			self.strata = None
			self.estimates = Estimators()
		elif self.cutoff == 0:
			pass
		else:
			raise ValueError('Invalid cutoff.')


	def trim_s(self):

		"""
		Trims data based on propensity score using the cutoff
		selection algorithm suggested by [1]_.
		
		This method should only be executed after the propensity score
		has been estimated.

		References
		----------
		.. [1] Crump, R., Hotz, V., Imbens, G., & Mitnik, O. (2009).
			Dealing with Limited Overlap in Estimation of
			Average Treatment Effects. Biometrika, 96, 187-199.
		"""

		pscore = self.raw_data['pscore']
		g = 1.0/(pscore*(1-pscore))  # 1 over Bernoulli variance

		self.cutoff = select_cutoff(g)
		self.trim()


	def stratify(self):

		"""
		Stratifies the sample based on propensity score.
		
		By default the sample is divided into five equal-sized bins.
		The number of bins can be set by modifying the object
		attribute named blocks. Alternatively, custom-sized bins can
		be created by setting blocks equal to a sorted list of numbers
		between 0 and 1 indicating the bin boundaries.

		This method should only be executed after the propensity score
		has been estimated.
		"""

		Y, D, X = self.raw_data['Y'], self.raw_data['D'], self.raw_data['X']
		pscore = self.raw_data['pscore']

		if isinstance(self.blocks, int):
			blocks = split_equal_bins(pscore, self.blocks)
		else:
			blocks = self.blocks[:]  # make a copy; should be sorted
			blocks[0] = 0  # avoids always dropping 1st unit

		def subset(p_low, p_high):
			return (p_low < pscore) & (pscore <= p_high)
		subsets = [subset(*ps) for ps in zip(blocks, blocks[1:])]
		strata = [CausalModel(Y[s], D[s], X[s]) for s in subsets]
		self.strata = Strata(strata, subsets, pscore)


	def stratify_s(self):

		"""
		Stratifies the sample based on propensity score using the
		bin selection procedure suggested by [1]_.

		The bin selection algorithm is based on a sequence of
		two-sample t tests performed on the log-odds ratio.

		This method should only be executed after the propensity score
		has been estimated.

		References
		----------
		.. [1] Imbens, G. & Rubin, D. (2015). Causal Inference in
			Statistics, Social, and Biomedical Sciences: An
			Introduction.
		"""

		pscore_order = self.raw_data['pscore'].argsort()
		pscore = self.raw_data['pscore'][pscore_order]
		D = self.raw_data['D'][pscore_order]
		logodds = np.log(pscore / (1-pscore))
		K = self.raw_data['K']

		blocks_uniq = set(select_blocks(pscore, logodds, D, K, 0, 1))
		self.blocks = sorted(blocks_uniq)
		self.stratify()


	def est_via_ols(self, adj=2):

		"""
		Estimates average treatment effects using least squares.

		Parameters
		----------
		adj: int (0, 1, or 2)
			Indicates how covariate adjustments are to be
			performed. Set adj = 0 to not include any
			covariates.  Set adj = 1 to include treatment
			indicator D and covariates X separately. Set
			adj = 2 to additionally include interaction
			terms between D and X. Defaults to 2.
		"""

		self.estimates['ols'] = OLS(self.raw_data, adj)


	def est_via_blocking(self, adj=1):

		"""
		Estimates average treatment effects using regression within
		blocks.
		
		This method should only be executed after the sample has been
		stratified.

		Parameters
		----------
		adj: int (0, 1, or 2)
			Indicates how covariate adjustments are to be
			performed for each within-bin regression.
			Set adj = 0 to not include any covariates.
			Set adj = 1 to include treatment indicator D
			and covariates X separately. Set adj = 2 to
			additionally include interaction terms between
			D and X. Defaults to 1.
		"""

		self.estimates['blocking'] = Blocking(self.strata, adj)


	def est_via_weighting(self):

		"""
		Estimates average treatment effects using doubly-robust
		version of the Horvitz-Thompson weighting estimator.
		"""

		self.estimates['weighting'] = Weighting(self.raw_data)


	def est_via_matching(self, weights='inv', matches=1, bias_adj=False):

		"""
		Estimates average treatment effects using nearest-
		neighborhood matching.

		Matching is done with replacement. Method supports multiple
		matching. Correcting bias that arise due to imperfect matches
		is also supported. For details on methodology, see [1]_.

		Parameters
		----------
		weights: str or positive definite square matrix
			Specifies weighting matrix used in computing
			distance measures. Defaults to string 'inv',
			which does inverse variance weighting. String
			'maha' gives the weighting matrix used in the
			Mahalanobis metric.
		matches: int
			Number of matches to use for each subject.
		bias_adj: bool
			Specifies whether bias adjustments should be
			attempted.

		References
		----------
		.. [1] Imbens, G. & Rubin, D. (2015). Causal Inference in
			Statistics, Social, and Biomedical Sciences: An
			Introduction.
		"""

		X, K = self.raw_data['X'], self.raw_data['K']
		X_c, X_t = self.raw_data['X_c'], self.raw_data['X_t']

		if weights == 'inv':
			W = 1/X.var(0)
		elif weights == 'maha':
			V_c = np.cov(X_c, rowvar=False, ddof=0)
			V_t = np.cov(X_t, rowvar=False, ddof=0)
			if K == 1:
				W = 1/np.array([[(V_c+V_t)/2]])  # matrix form
			else:
				W = np.linalg.inv((V_c+V_t)/2)
		else:
			W = weights

		self.estimates['matching'] = Matching(self.raw_data, W,
		                                      matches, bias_adj)


	def _post_pscore_init(self):

		self.cutoff = 0.1
		self.blocks = 5


def parse_lin_terms(K, lin):

	if lin is None:
		return []
	elif lin == 'all':
		return range(K)
	else:
		return lin


def parse_qua_terms(K, qua):

	if qua is None:
		return []
	elif qua == 'all':
		return list(combinations_with_replacement(range(K), 2))
	else:
		return qua


def sumlessthan(g, sorted_g, cumsum):

	deduped_values = dict(zip(sorted_g, cumsum))

	return np.array([deduped_values[x] for x in g])


def select_cutoff(g):

	if g.max() <= 2*g.mean():
		cutoff = 0
	else:
		sorted_g = np.sort(g)
		cumsum_1 = range(1, len(g)+1)
		LHS = g * sumlessthan(g, sorted_g, cumsum_1)
		cumsum_g = np.cumsum(sorted_g)
		RHS = 2 * sumlessthan(g, sorted_g, cumsum_g)
		gamma = np.max(g[LHS <= RHS])
		cutoff = 0.5 - np.sqrt(0.25 - 1./gamma)

	return cutoff


def split_equal_bins(pscore, blocks):

	q = np.linspace(0, 100, blocks+1)[1:-1]  # q as in qth centiles
	centiles = [np.percentile(pscore, x) for x in q]

	return [0] + centiles + [1]


def calc_tstat(sample_c, sample_t):

	N_c = sample_c.shape[0]
	N_t = sample_t.shape[0]
	var_c = sample_c.var(ddof=1)
	var_t = sample_t.var(ddof=1)

	return (sample_t.mean()-sample_c.mean()) / np.sqrt(var_c/N_c+var_t/N_t)


def calc_sample_sizes(D):

	N = D.shape[0]
	mid_index  = N // 2
	
	Nleft = mid_index
	Nleft_t = D[:mid_index].sum()
	Nleft_c = Nleft - Nleft_t

	Nright = N - Nleft
	Nright_t = D[mid_index:].sum()
	Nright_c = Nright - Nright_t

	return (Nleft_c, Nleft_t, Nright_c, Nright_t)


def select_blocks(pscore, logodds, D, K, p_low, p_high):

	scope = (pscore >= p_low) & (pscore <= p_high)
	c, t = (scope & (D==0)), (scope & (D==1))

	Nleft_c, Nleft_t, Nright_c, Nright_t = calc_sample_sizes(D[scope])
	if min(Nleft_c, Nleft_t, Nright_c, Nright_t) < K+1:
		return [p_low, p_high]

	tstat = calc_tstat(logodds[c], logodds[t])
	if tstat <= 1.96:
		return [p_low, p_high]

	low = pscore[scope][0]
	mid = pscore[scope][scope.sum() // 2]
	high = pscore[scope][-1]

	return select_blocks(pscore, logodds, D, K, low, mid) + \
	       select_blocks(pscore, logodds, D, K, mid, high)

