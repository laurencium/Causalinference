from __future__ import division
import numpy as np
import scipy.linalg
from itertools import chain

from .basic import Basic
from .strata import Stratum, Strata
from .propensity import Propensity, PropensitySelect
from .estimates import Estimates
from utils.tools import remove


class CausalModel(Basic):


	def __init__(self, Y, D, X):

		super(CausalModel, self).__init__(Y, D, X)
		self.est = Estimates()
		self._Y_old, self._D_old, self._X_old = Y, D, X


	@property
	def Xvar(self):

		try:
			return self._Xvar
		except AttributeError:
			self._Xvar = self.X.var(0)
			return self._Xvar


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
				Column numbers (one-based) of the original
				covariate matrix X to include linearly. Defaults
				to the string 'all', which uses whole covariate
				matrix.
			qua: list
				Column numbers (one-based) of the original
				covariate matrix X to include as quadratic
				terms. E.g., [1,3] will include squares of the
				1st and 3rd columns, and the product of these
				two columns.  Default is to not include any
				quadratic terms.
		"""

		self.pscore = Propensity(self.D, self.X, lin, qua)
		self._post_pscore_init()


	def propensity_s(self, lin_B=[], C_lin=1, C_qua=2.71):

		"""
		Estimates via logit the propensity score using the covariate
		selection algorithm of Imbens and Rubin (2015).

		Expected args
		-------------
			lin_B: list
				Column numbers (one-based) of the original
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

		self.pscore = PropensitySelect(self.D, self.X,
		                               lin_B, C_lin, C_qua)
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
		self.est = Estimates()


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

		ate = np.sum([s.N/self.N*s.within for s in self.strata])
		att = np.sum([s.N_t/self.N_t*s.within for s in self.strata])
		atc = np.sum([s.N_c/self.N_c*s.within for s in self.strata])
		self.est._add(ate, att, atc, 'blocking', self)


	def _compute_blocking_se(self):

		"""
		Computes standard errors for average treatment effects
		estimated via regression within blocks.
		"""

		self._check_prereq('strata')

		wvar = [(s.N/self.N)**2 * s.se**2 for s in self.strata] 
		wvar_t = [(s.N_t/self.N_t)**2 * s.se**2 for s in self.strata]
		wvar_c = [(s.N_c/self.N_c)**2 * s.se**2 for s in self.strata]
		ate_se = np.sqrt(np.array(wvar).sum())
		att_se = np.sqrt(np.array(wvar_t).sum())
		atc_se = np.sqrt(np.array(wvar_c).sum())

		return (ate_se, att_se, atc_se)


	def _norm(self, dX, W):

		"""
		Calculates vector of norms given weighting matrix W.

		Expected args
		-------------
			dX: array-like
				Matrix of covariate differences.
			W: string or matrix, ndarray
				Weighting matrix to be used in norm calcuation.
				Acceptable values are string 'inv' for inverse
				variance weighting, or any arbitrary K-by-K
				matrix.

		Returns
		-------
			Vector of distance measures.
		"""

		if W == 'inv':
			return (dX**2 / self.Xvar).sum(axis=1)
		else:
			return (dX.dot(W)*dX).sum(axis=1)


	def _msmallest_w_ties(self, x, m):

		"""
		Finds indices of the m smallest entries in an array. Ties are
		included, so the number of indices can be greater than m.
		Algorithm is of order O(n).

		Expected args
		-------------
			x: array-like
				Array of numbers to find m smallest entries for.
			m: integer
				Number of smallest entries to find.

		Returns
		-------
			List of indices of smallest entries.
		"""

		# partition around (m+1)th order stat
		par_indx = np.argpartition(x, m)
		
		if x[par_indx[:m]].max() < x[par_indx[m]]:  # m < (m+1)th
			return list(par_indx[:m])
		elif x[par_indx[m]] < x[par_indx[m+1:]].min():  # m+1 < (m+2)th
			return list(par_indx[:m+1])
		else:  # mth = (m+1)th = (m+2)th, so increment and recurse
			return self._msmallest_w_ties(x, m+2)


	def _make_matches(self, X, X_m, W, m):

		"""
		Performs nearest-neigborhood matching using specified weighting
		matrix in measuring distance. Ties are included, so the number
		of matches for a given unit can be greater than m.

		Expected args
		-------------
			X: matrix, ndarray
				Observations to find matches for.
			X_m: matrix, ndarray
				Pool of potential matches.
			W: string or matrix, ndarray
				Weighting matrix to be used in norm calcuation.
				Acceptable values are string 'inv' for inverse
				variance weighting, or any arbitrary K-by-K
				matrix.
			m: integer
				The number of units to match to a given subject.

		Returns
		-------
			List of matched indices.
		"""

		m_indx = [None] * X.shape[0]

		for i in xrange(X.shape[0]):
			m_indx[i] = self._msmallest_w_ties(self._norm(X_m-X[i],
			                                              W), m)

		return m_indx


	def _bias(self, m_indx, Y_m, X_m, X):

		"""
		Estimates bias resulting from imperfect matches using least
		squares.  When estimating ATT, regression should use control
		units. When estimating ATC, regression should use treated units.

		Expected args
		-------------
			m_indx: list
				Index of indices of matched units.
			Y_m: array-like
				Vector of outcomes to regress.
			X_m: matrix, ndarray
				Covariate matrix to regress on.
			X: matrix, ndarray
				Covariate matrix of subjects under study.

		Returns
		-------
			Vector of estimated biases.
		"""

		flat_indx = list(chain.from_iterable(m_indx))

		X_m1 = np.empty((len(flat_indx), X_m.shape[1]+1))
		X_m1[:,0] = 1
		X_m1[:,1:] = X_m[flat_indx]
		beta = np.linalg.lstsq(X_m1, Y_m[flat_indx])[0]

		return [np.dot(X[i]-X_m[m_indx[i]].mean(0),
		               beta[1:]) for i in xrange(X.shape[0])]
	

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

		References
		----------
			Imbens, G. & Rubin, D. (2015). Causal Inference in
				Statistics, Social, and Biomedical Sciences:
				An Introduction.

		Expected args
		-------------
			wmat: string or matrix, ndarray
				Weighting matrix to be used in norm calcuation.
				Acceptable values are None (inverse variance,
				default), string 'maha' for Mahalanobis	metric,
				or any arbitrary k-by-k matrix.
			m: integer
				The number of units to match to a given
				subject. Defaults to 1.
			xbias: Boolean
				Correct bias resulting from imperfect matches or
				not; defaults to no correction.
		"""

		if wmat == 'maha':
			self._wmat = np.linalg.inv(np.cov(self.X, rowvar=False))
		else:
			self._wmat = wmat
		self._m = m

		self._m_indx_t = self._make_matches(self.X_t, self.X_c,
		                                    self._wmat, self._m)
		self._m_indx_c = self._make_matches(self.X_c, self.X_t,
		                                    self._wmat, self._m)

		self.ITT = np.empty(self.N)
		self.ITT[self.D==1] = self.Y_t - [self.Y_c[self._m_indx_t[i]].mean() for i in xrange(self.N_t)]
		self.ITT[self.D==0] = [self.Y_t[self._m_indx_c[i]].mean() for i in xrange(self.N_c)] - self.Y_c

		if xbias:
			self.ITT[self.D==1] -= self._bias(self._m_indx_t, self.Y_c, self.X_c, self.X_t)
			self.ITT[self.D==0] += self._bias(self._m_indx_c, self.Y_t, self.X_t, self.X_c)

		ate = self.ITT.mean()
		att = self.ITT[self.D==1].mean()
		atc = self.ITT[self.D==0].mean()
		self.est._add(ate, att, atc, 'matching', self)


	def _compute_condvar(self, W, m):

		"""
		Computes unit-level conditional variances. Estimation is done by
		matching treated units with treated units, control units with
		control units, and then calculating sample variances among the
		matches.

		Expected args
		-------------
			W: string or matrix, ndarray
				Weighting matrix to be used in norm calcuation.
				Acceptable values are string 'inv' for inverse
				variance weighting, or any arbitrary K-by-K
				matrix.
			m: integer
				The number of units to match to a given subject.
		"""

		# m+1 since we include the unit itself in matching pool as well
		m_indx_t = self._make_matches(self.X_t, self.X_t, W, m+1)
		m_indx_c = self._make_matches(self.X_c, self.X_c, W, m+1)

		self._condvar = np.empty(self.N)
		self._condvar[self.D==1] = [self.Y_t[m_indx_t[i]].var(ddof=1) for i in xrange(self.N_t)]
		self._condvar[self.D==0] = [self.Y_c[m_indx_c[i]].var(ddof=1) for i in xrange(self.N_c)]


	def _count_matches(self, m_indx_t, m_indx_c):

		"""
		Calculates each unit's contribution in being used as a matching
		unit.

		Expected args
		-------------
			m_indx_t: list
				List of indices of control units that are
				matched to each treated	unit. 
			m_indx_c:
				List of indices of treated units that are
				matched to each control unit.

		Returns
		-------
			Vector containing each unit's contribution in matching.
		"""

		count = np.zeros(self.N)
		treated = np.nonzero(self.D)[0]
		control = np.nonzero(self.D==0)[0]
		for i in xrange(self.N_c):
			M = len(m_indx_c[i])
			for j in xrange(M):
				count[treated[m_indx_c[i][j]]] += 1./M
		for i in xrange(self.N_t):
			M = len(m_indx_t[i])
			for j in xrange(M):
				count[control[m_indx_t[i][j]]] += 1./M

		return count


	def _compute_matching_se(self):

		self._compute_condvar(self._wmat, self._m)
		match_counts = self._count_matches(self._m_indx_t, self._m_indx_c)
		ate_se = np.sqrt(((1+match_counts)**2 * self._condvar).sum() / self.N**2)
		att_se = np.sqrt(((self.D - (1-self.D)*match_counts)**2 * self._condvar).sum() / self.N_t**2)
		atc_se = np.sqrt(((self.D*match_counts - (1-self.D))**2 * self._condvar).sum() / self.N_c**2)

		return (ate_se, att_se, atc_se)


	def _ols_predict(self, Y, X, X_new):

		"""
		Estimates linear regression model with least squares and project
		based on new input data.

		Expected args
		-------------
			Y: array-like
				Vector of observed outcomes.
			X: matrix, ndarray
				Matrix of covariates to regress on.
			X_new: matrix, ndarray
				Matrix of covariates used to generate
				predictions.

		Returns
		-------
			Vector of predicted values.
		"""

		Z = np.empty((X.shape[0], X.shape[1]+1))
		Z[:, 0] = 1  # constant term
		Z[:, 1:] = X
		beta = np.linalg.lstsq(Z, Y)[0]

		return beta[0] + X_new.dot(beta[1:])


	def weighting(self):

		"""
		Computes ATE using the Horvitz-Thompson weighting estimator
		modified to incorporate covariates.

		References
		----------
			Lunceford, J. K., & Davidian, M. (2004). Stratification
				and weighting via the propensity score in
				estimation of causal treatment effects: a
				comparative study. Statistics in Medicine.
		"""

		self._check_prereq('pscore')

		phat = self.pscore['fitted']
		Yhat_t = self._ols_predict(self.Y_t, self.X_t, self.X)
		Yhat_c = self._ols_predict(self.Y_c, self.X_c, self.X)
		summand = (self.D-phat) * (self.Y - (1-phat)*Yhat_t - \
		          phat*Yhat_c) / (phat*(1-phat))

		ate = summand.mean()
		att = summand[self.D==1].mean()
		atc = summand[self.D==0].mean()
		ate_se = np.sqrt(summand.var()/self.N)
		att_se = np.sqrt(summand[self.D==1].var()/self.N_t)
		atc_se = np.sqrt(summand[self.D==0].var()/self.N_c)

		self.est._add(ate, att, atc, 'weighting', self)
		self.est['weighting']._add_se(ate_se, att_se, atc_se)


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

		Equivalently, we can recover ATE directly from the regression
			Y = b_0 + b_1 * D + b_2 * D(X-mean(X)) + b_3 * X + e.
		The estimated coefficient b_1 will then be numerically identical
		to the ATE estimate obtained from the first method. ATT can then
		be computed by
			b_1 + b_2 * (mean(X_t)-mean(X)),
		and analogously for ATC. The advantage of this single regression
		approach is that the matrices required for heteroskedasticity-
		robust covariance matrix estimation can be obtained
		conveniently. This is the apporach used.

		Least squares estimates are computed via QR factorization. The
		design matrix and the R matrix are stored in case standard
		errors need to be computed later.
		"""

		Xmean = self.X.mean(0)
		self._Z = np.empty((self.N, 2+2*self.K))  # create design matrix
		self._Z[:,0] = 1  # constant term
		self._Z[:,1] = self.D
		self._Z[:,2:2+self.K] = self.D[:,None]*(self.X-Xmean)
		self._Z[:,-self.K:] = self.X

		Q, self._R = np.linalg.qr(self._Z)
		self._olscoef = scipy.linalg.solve_triangular(self._R,
		                                               Q.T.dot(self.Y))

		ate = self._olscoef[1]
		att = self._olscoef[1] + np.dot(self.X_t.mean(0)-Xmean,
		                                self._olscoef[2:2+self.K])
		atc = self._olscoef[1] + np.dot(self.X_c.mean(0)-Xmean,
		                                self._olscoef[2:2+self.K])
		self.est._add(ate, att, atc, 'ols', self)


	def _compute_ols_se(self):
	
		"""
		Computes standard errors for OLS estimates of ATE, ATT, and ATC.

		If Z denotes the design matrix (i.e., covariates, treatment
		indicator, product of the two, and a column of ones) and u
		denotes the vector of least squares residual, then the variance
		estimator can be found by computing White's heteroskedasticity-
		robust covariance matrix:
			inv(Z'Z) Z'diag(u^2)Z inv(Z'Z).
		The diagonal entry corresponding to the treatment indicator of
		this matrix is the appropriate variance estimate for ATE.
		Variance estimates for ATT and ATC are appropriately weighted
		sums of entries of the above matrix.
		"""

		Xmean = self.X.mean(0)
		u = self.Y - self._Z.dot(self._olscoef)
		A = np.linalg.inv(np.dot(self._R.T, self._R))
		# select columns for D, D*dX from A
		B = np.dot(u[:,None]*self._Z, A[:,1:2+self.K])  
		covmat = np.dot(B.T, B)

		ate_se = np.sqrt(covmat[0,0])
		C = np.empty(self.K+1); C[0] = 1
		C[1:] = self.X_t.mean(0)-Xmean
		att_se = np.sqrt(C.dot(covmat).dot(C))
		C[1:] = self.X_c.mean(0)-Xmean
		atc_se = np.sqrt(C.dot(covmat).dot(C))

		return (ate_se, att_se, atc_se)


	def _compute_se(self, method):

		"""
		Wrapper function that calls requested standard-error-computing
		function.

		Expected args
		-------------
			method: string
				One of 'ols', 'blocking', or 'matching'; used
				to determine which function to call to compute
				standard errors.

		Returns
		-------
			3-tuple of standard errors for ATE, ATT, and ATC,
			respectively.
		"""

		if method == 'ols':
			return self._compute_ols_se()
		elif method == 'blocking':
			return self._compute_blocking_se()
		elif method == 'matching':
			return self._compute_matching_se()

