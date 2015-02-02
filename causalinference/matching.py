from itertools import chain


from .estimators import Estimator


class Matching(Estimator):


	def __init__(self, wmat, m, xbias, data):

		N, N_c, N_t = data.N, data.N_c, data.N_t
		X, X_c, X_t = data.X, data.X_c, data.X_t
		Y, Y_c, Y_t = data.Y, data.Y_c, data.Y_t
		c, t = (data.D==0), (data.D==1)

		self._inv = (wmat == 'inv')
		if wmat == 'maha':
			self._wmat = np.linalg.inv(np.cov(X, False))
		elif wmat == 'inv':
			self._wmat = X.var(0)
		else:
			self._wmat = wmat

		self._m = m
		self._xbias = xbias

		self._m_indx_t = self._make_matches(X_t, X_c)
		self._m_indx_c = self._make_matches(X_c, X_t)

		self._ITT = np.empty(N)
		Yhat_c = [Y_c[self._m_indx_t[i]].mean() for i in xrange(N_t)]
		self._ITT[t] = Y_t - Yhat_c
		Yhat_t = [Y_t[self._m_indx_c[i]].mean() for i in xrange(N_c)]
		self._ITT[c] = Yhat_t - Y_c

		if xbias:
			self._ITT[t] -= self._bias(self._m_indx_t,
			                           Y_c, X_c, X_t)
			self._ITT[c] += self._bias(self._m_indx_c,
			                           Y_t, X_t, X_c)

		ate = self._ITT.mean()
		att = self._ITT[t].mean()
		atc = self._ITT[c].mean()

		super(Matching, self).__init__(ate, att, atc, 'matching', self)


	def _norm(self, dX):

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

		if self._inv:
			return (dX**2 / self._wmat).sum(1)
		else:
			return (dX.dot(self._wmat)*dX).sum(1)


	def _msmallest(self, x):

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

		m = self._m

		# partition around (m+1)th order stat
		par_indx = np.argpartition(x, m)
		
		if x[par_indx[:m]].max() < x[par_indx[m]]:  # m < (m+1)th
			return list(par_indx[:m])
		elif x[par_indx[m]] < x[par_indx[m+1:]].min():  # m+1 < (m+2)th
			return list(par_indx[:m+1])
		else:  # mth = (m+1)th = (m+2)th, so increment and recurse
			return self._msmallest(x, m+2)


	def _make_matches(self, X, X_m):

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
			norm = self._norm(X_m-X[i])
			m_indx[i] = self._msmallest(norm)

		return m_indx


	def _bias(self, m_indx, Y_m, X_m, X):

		"""
		Estimates bias resulting from imperfect matches using least
		squares. When estimating ATT, regression should use control
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

