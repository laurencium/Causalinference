import numpy as np
from itertools import chain

from base import Estimator


class Matching(Estimator):

	"""
	Dictionary-like class containing treatment effect estimates. Standard
	errors are only computed when needed.
	"""

	def __init__(self, wmat, m, xbias, model):

		self._model = model
		self._inv = (wmat == 'inv')
		if wmat == 'maha':
			self._wmat = np.linalg.inv(np.cov(self._model.X,
			                                  rowvar=False))
		elif wmat == 'inv':
			self._wmat = self._model.X.var(0)
		else:
			self._wmat = wmat
		self._m = m
		self._xbias = xbias
		super(Matching, self).__init__()


	def _norm(self, dX):

		"""
		Calculates vector of norms using previously specified weighting
		matrix.

		Expected args
		-------------
			dX: array-like
				Matrix of covariate differences.

		Returns
		-------
			Vector of distance measures.
		"""

		if self._inv:
			return (dX**2 / self._wmat).sum(1)
		else:
			return (dX.dot(self._wmat)*dX).sum(1)


	def _msmallest(self, x, m):

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
		par_idx = np.argpartition(x, m)
		
		if x[par_idx[:m]].max() < x[par_idx[m]]:  # m < (m+1)th
			return list(par_idx[:m])
		elif x[par_idx[m]] < x[par_idx[m+1:]].min():  # m+1 < (m+2)th
			return list(par_idx[:m+1])
		else:  # mth = (m+1)th = (m+2)th, so increment and recurse
			return self._msmallest(x, m+2)


	def _make_matches(self, X, X_m, m):

		"""
		Performs nearest-neigborhood matching using previously specified
		weighting matrix in measuring distance. Ties are included, so
		the number of matches for a given unit can be greater than m.

		Expected args
		-------------
			X: matrix, ndarray
				Observations to find matches for.
			X_m: matrix, ndarray
				Pool of potential matches.
			m: integer
				The number of units to match to a given subject.

		Returns
		-------
			List of matched indices.
		"""

		m_idx = [None] * X.shape[0]

		for i in xrange(X.shape[0]):
			norm = self._norm(X_m-X[i])
			m_idx[i] = self._msmallest(norm, m)

		return m_idx


	def _bias(self, m_idx, Y_m, X_m, X):

		"""
		Estimates bias resulting from imperfect matches using least
		squares. When estimating ATT, regression should use control
		units. When estimating ATC, regression should use treated units.
		See Imbens and Rubin (2015) for details.

		Expected args
		-------------
			m_idx: list
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

		References
		----------
			Imbens, G. & Rubin, D. (2015). Causal Inference in
				Statistics, Social, and Biomedical Sciences:
				An Introduction.
		"""

		flat_idx = list(chain.from_iterable(m_idx))

		X_m1 = np.empty((len(flat_idx), X_m.shape[1]+1))
		X_m1[:,0] = 1
		X_m1[:,1:] = X_m[flat_idx]
		beta = np.linalg.lstsq(X_m1, Y_m[flat_idx])[0]

		return [np.dot(X[i]-X_m[m_idx[i]].mean(0),
		               beta[1:]) for i in xrange(X.shape[0])]
	

	def _compute_condvar(self):

		"""
		Computes unit-level conditional variances. Estimation is done by
		matching treated units with treated units, control units with
		control units, and then calculating sample variances among the
		matches.
		"""

		N, N_c, N_t = self._model.N, self._model.N_c, self._model.N_t
		X, X_c, X_t = self._model.X, self._model.X_c, self._model.X_t
		Y, Y_c, Y_t = self._model.Y, self._model.Y_c, self._model.Y_t
		c, t = self._model.controls, self._model.treated

		# m+1 since we include the unit itself in matching pool as well
		idx_c = self._make_matches(X_c, X_c, self._m+1)
		idx_t = self._make_matches(X_t, X_t, self._m+1)

		condvar = np.empty(N)
		condvar[c] = [Y_c[idx_c[i]].var(ddof=1) for i in xrange(N_c)]
		condvar[t] = [Y_t[idx_t[i]].var(ddof=1) for i in xrange(N_t)]

		return condvar


	def _count_matches(self):

		"""
		Calculates each unit's contribution in being used as a matching
		unit.

		Returns
		-------
			Vector containing each unit's contribution in matching.
		"""

		N, N_c, N_t = self._model.N, self._model.N_c, self._model.N_t
		controls, treated = self._model.controls, self._model.treated

		count = np.zeros(N)
		c, t = np.nonzero(controls)[0], np.nonzero(treated)[0]
		for i in xrange(N_c):
			M = len(self._idx_c[i])
			for j in xrange(M):
				count[t[self._idx_c[i][j]]] += 1./M
		for i in xrange(N_t):
			M = len(self._idx_t[i])
			for j in xrange(M):
				count[c[self._idx_t[i][j]]] += 1./M

		return count


	def _compute_est(self):

		"""
		Estimates average treatment effects using matching with
		replacement.

		Returns
		-------
			3-tuple of ATE, ATT, and ATC estimates, respectively.

		References
		----------
			Imbens, G. & Rubin, D. (2015). Causal Inference in
				Statistics, Social, and Biomedical Sciences:
				An Introduction.
		"""

		N, N_c, N_t = self._model.N, self._model.N_c, self._model.N_t
		X, X_c, X_t = self._model.X, self._model.X_c, self._model.X_t
		Y, Y_c, Y_t = self._model.Y, self._model.Y_c, self._model.Y_t
		c, t = self._model.controls, self._model.treated

		self._idx_c = self._make_matches(X_c, X_t, self._m)
		self._idx_t = self._make_matches(X_t, X_c, self._m)

		self._ITT = np.empty(N)
		Yhat_t = [Y_t[self._idx_c[i]].mean() for i in xrange(N_c)]
		self._ITT[c] = Yhat_t - Y_c
		Yhat_c = [Y_c[self._idx_t[i]].mean() for i in xrange(N_t)]
		self._ITT[t] = Y_t - Yhat_c

		if self._xbias:
			self._ITT[c] += self._bias(self._idx_c, Y_t, X_t, X_c)
			self._ITT[t] -= self._bias(self._idx_t, Y_c, X_c, X_t)

		ate = self._ITT.mean()
		att = self._ITT[t].mean()
		atc = self._ITT[c].mean()

		return (ate, att, atc)


	def _compute_se(self):

		"""
		Computes standard errors for average treatment effects using
		the general variance estimator in Imbens and Rubin (2015).
		This involves first computing estimates of unit-level
		conditional variances via matching, and then taking an
		appropriately weighted sum of these estimates.

		Returns
		-------
			3-tuple of ATE, ATT, and ATC standard error estimates,
			respectively.

		References
		----------
			Imbens, G. & Rubin, D. (2015). Causal Inference in
				Statistics, Social, and Biomedical Sciences:
				An Introduction.
		"""

		N, N_c, N_t = self._model.N, self._model.N_c, self._model.N_t
		D = self._model.D

		condvar = self._compute_condvar()
		M = self._count_matches()
		ate_se = np.sqrt(((1+M)**2 * condvar).sum() / N**2)
		att_se = np.sqrt(((D - (1-D)*M)**2 * condvar).sum() / N_t**2)
		atc_se = np.sqrt(((D*M - (1-D))**2 * condvar).sum() / N_c**2)

		return (ate_se, att_se, atc_se)

