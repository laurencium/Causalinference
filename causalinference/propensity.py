import numpy as np
from scipy.optimize import fmin_bfgs
from itertools import combinations_with_replacement


class Propensity(object):

	"""
	Dictionary-like class containing propensity score data, including
	estimated logistic regression coefficients, predicted propensity score,
	maximized log-likelihood, and lists of the linear and quadratic terms
	that are included in the regression.
	"""

	def __init__(self, D, X, lin, qua):

		if lin == 'all':
			lin = range(X.shape[1])
		else:
			lin = self._change_base(lin, base=0)
		qua = self._change_base(qua, pair=True, base=0)

		mat = self._form_matrix(X, lin, qua)
		self._dict = self._compute_pscore(D, mat) 
		self._dict['lin'], self._dict['qua'] = lin, qua


	def __getitem__(self, key):

		return self._dict[key]


	def __setitem__(self, key, value):

		if key == 'fitted':
			self._dict[key] = value
		else:
			raise TypeError("'" + self.__class__.__name__ +
			                "' object does not support item " +
					"assignment")


	def __str__(self):

		return 'Propensity class string placeholder.'


	def keys(self):

		return self._dict.keys()


	def _sigmoid(self, x):
	
		"""
		Computes 1/(1+exp(-x)) for input x, to be used in maximum
		likelihood estimation of propensity score.

		Expected args
		-------------
			x: array-like

		Returns
		-------
			Vector or scalar 1/(1+exp(-x)), depending on input x.
		"""

		return 1/(1+np.exp(-x))


	def _log1exp(self, x):

		"""
		Computes log(1+exp(-x)) for input x, to be used in maximum
		likelihood estimation of propensity score.

		Expected args
		-------------
			x: array-like

		Returns
		-------
			Vector or scalar log(1+exp(-x)), depending on input x.
		"""

		return np.log(1 + np.exp(-x))


	def _neg_loglike(self, beta, X_c, X_t):

		"""
		Computes the negative of the log likelihood function for logit,
		to be used in maximum likelihood estimation of propensity score.
		Negative because SciPy optimizier does minimization only.

		Expected args
		-------------
			beta: array-like
				Logisitic regression parameters to maximize
				over.
			X_c: matrix, ndarray
				Covariate matrix of the control units.
			X_t: matrix, ndarray
				Covariate matrix of the treated units.

		Returns
		-------
			Negative log likelihood evaluated at input values.
		"""

		return self._log1exp(X_t.dot(beta)).sum() + \
		       self._log1exp(-X_c.dot(beta)).sum()


	def _neg_gradient(self, beta, X_c, X_t):

		"""
		Computes the negative of the gradient of the log likelihood
		function for logit, to be used in maximum likelihood estimation
		of propensity score. Negative because SciPy optimizier does
		minimization only.

		Expected args
		-------------
			beta: array-like
				Logisitic regression parameters to maximize over.
			X_c: matrix, ndarray
				Covariate matrix of the control units.
			X_t: matrix, ndarray
				Covariate matrix of the treated units.

		Returns
		-------
			Negative gradient of log likelihood function evaluated
			at input values.
		"""

		return (self._sigmoid(X_c.dot(beta))*X_c.T).sum(1) - \
		       (self._sigmoid(-X_t.dot(beta))*X_t.T).sum(1)


	def _compute_pscore(self, D, X):

		"""
		Estimates via logit the propensity score based on input
		covariate matrix X. Uses BFGS algorithm for optimization.

		Expected args
		-------------
			X: matrix, ndarray
				Covariate matrix to estimate propensity score on.

		Returns
		-------
			pscore: dict containing
				'coef': Estimated coefficients.
				'loglike': Maximized log-likelihood value.
				'fitted': Vector of estimated propensity scores.
		"""

		X_c, X_t = X[D==0], X[D==1]
		N, K = X.shape

		neg_loglike = lambda x: self._neg_loglike(x, X_c, X_t)
		neg_gradient = lambda x: self._neg_gradient(x, X_c, X_t)

		logit = fmin_bfgs(neg_loglike, np.zeros(K), neg_gradient,
		                  full_output=True, disp=False)

		pscore = {}
		pscore['coef'], pscore['loglike'] = logit[0], -logit[1]
		pscore['fitted'] = np.empty(N)
		pscore['fitted'][D==0] = self._sigmoid(X_c.dot(pscore['coef']))
		pscore['fitted'][D==1] = self._sigmoid(X_t.dot(pscore['coef']))

		return pscore


	def _form_matrix(self, X, lin, qua):

		"""
		Forms covariate matrix for use in propensity score estimation,
		based on requirements on linear and quadratic terms.

		Expected args
		-------------
			lin: list
				Column numbers (one-based) of the original
				covariate matrix to include linearly.
			qua: list
				Tuples indicating which columns of the original
				covariate matrix to multiply and include. E.g.,
				[(1,1), (2,3)] indicates squaring the 1st column
				and including the product of the 2nd and 3rd
				columns.

		Returns
		-------
			mat: matrix, ndarray
				Covariate matrix formed based on requirements on
				linear and quadratic terms.
		"""

		mat = np.empty((X.shape[0], 1+len(lin)+len(qua)))

		mat[:, 0] = 1  # constant term
		current_col = 1
		if lin:
			mat[:, current_col:current_col+len(lin)] = X[:, lin]
			current_col += len(lin)
		for term in qua:
			mat[:, current_col] = X[:, term[0]] * X[:, term[1]]
			current_col += 1

		return mat


	def _change_base(self, l, pair=False, base=0):

		"""
		Changes input index to zero or one-based.

		Expected args
		-------------
			l: list
				List of numbers or pairs of numbers.
			pair: Boolean
				Anticipates list of pairs if True. Defaults to
				False.
			base: integer
				Converts to zero-based if 0, one-based if 1.

		Returns
		-------
			Input index with base changed.
		"""

		offset = 2*base - 1
		if pair:
			return [(p[0]+offset, p[1]+offset) for p in l]
		else:
			return [e+offset for e in l]


class PropensitySelect(Propensity):

	"""
	Dictionary-like class containing propensity score data, including
	estimated logistic regression coefficients, predicted propensity score,
	maximized log-likelihood, and lists of the linear and quadratic terms
	that are included in the regression.
	"""

	def __init__(self, D, X, lin_B, C_lin, C_qua):

		lin_B = self._change_base(lin_B, base=0)
		if C_lin == np.inf:  # inf threshold, so include basic only
			lin = lin_B
		elif C_lin == 0:  # min threshold, so include everything
			lin = range(X.shape[1])
		else:
			pot = list(set(xrange(X.shape[1])) - set(lin_B))
			lin = self._select_terms(D, X, lin_B, pot, C_lin)

		if C_qua == np.inf:  # inf threshold, so include nothing
			qua = []
		elif C_qua == 0:  # min threshold, so include everything
			qua = list(combinations_with_replacement(lin, 2))
		else:
			pot = list(combinations_with_replacement(lin, 2))
			qua = self._select_terms(D, X, [], pot, C_qua, lin)

		mat = self._form_matrix(X, lin, qua)
		self._dict = self._compute_pscore(D, mat)
		self._dict['lin'], self._dict['qua'] = lin, qua
		

	def __str__(self):

		return 'PropensitySelect class string placeholder.'


	def _select_terms(self, D, X, cur, pot, crit, lin=[]):
	
		"""
		Estimates via logit the propensity score using Imbens and
		Rubin's covariate selection algorithm.

		Expected args
		-------------
			cur: list
				List containing terms that are currently
				included in the logistic regression.
			pot: list
				List containing candidate terms to be iterated
				through.
			crit: scalar
				Critical value used in likelihood ratio test
				to decide whether candidate terms should be
				included.
			lin: list
				List containing linear terms that have been
				decided on. If non-empty, then cur and pot
				should be containing candidate quadratic terms.
				If empty, then those two lists should be
				containing candidate linear terms.

		Returns
		-------
			List containing terms that the algorithm has settled on
			for inclusion.
		"""

		if not pot:
			return cur

		# calculate log-likelihood under null of no additional terms
		if not lin:  # lin is empty, so linear terms not yet decided
			mat = self._form_matrix(X, cur, [])
			ll_null = self._compute_pscore(D, mat)['loglike']
		else:  # lin is not empty, so linear terms are already fixed
			mat = self._form_matrix(X, lin, cur)
			ll_null = self._compute_pscore(D, mat)['loglike']

		# calculate LR stat after including each additional term
		lr = np.empty(len(pot))
		if not lin:
			for i in xrange(len(pot)):
				mat = self._form_matrix(X, cur+[pot[i]], [])
				ll = self._compute_pscore(D, mat)['loglike']
				lr[i] = 2*(ll - ll_null)
		else:
			for i in xrange(len(pot)):
				mat = self._form_matrix(X, lin, cur+[pot[i]])
				ll = self._compute_pscore(D, mat)['loglike']
				lr[i] = 2*(ll - ll_null)

		argmax = np.argmax(lr)
		if lr[argmax] < crit:
			return cur  # stop including additional terms
		else:
			# include new term and recurse on remaining
			new_term = pot.pop(argmax)
			return self._select_terms(D, X, cur+[new_term],
			                          pot, crit, lin)

