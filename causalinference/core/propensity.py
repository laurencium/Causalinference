import numpy as np
from scipy.optimize import fmin_bfgs
from itertools import combinations_with_replacement

from ..utils.tools import Printer


class Propensity(object):

	"""
	Dictionary-like class containing propensity score data, including
	estimated logistic regression coefficients, predicted propensity score,
	maximized log-likelihood, and lists of the linear and quadratic terms
	that are included in the regression.
	"""

	def __init__(self, lin, qua, model):

		self._model = model
		D, X = self._model.D, self._model.X

		if lin == 'all':
			lin = range(X.shape[1])

		mat = self._form_matrix(X, lin, qua)
		self._dict = self._compute_pscore(mat) 
		self._dict['lin'], self._dict['qua'] = lin, qua
		self._dict['se'] = None


	def __getitem__(self, key):

		if key == 'se' and self._dict['se'] is None:
			self._dict['se'] = self._compute_se()

		return self._dict[key]


	def __setitem__(self, key, value):

		if key == 'fitted':
			self._dict[key] = value
		else:
			raise TypeError("'" + self.__class__.__name__ +
			                "' object does not support item " +
					"assignment")


	def __str__(self):

		if self._dict['se'] is None:
			self._dict['se'] = self._compute_se()

		coef = self._dict['coef']
		se = self._dict['se']
		p = Printer()

		output = '\n'
		output += 'Estimated Parameters of Propensity Score\n\n'

		entries = ('', 'Coef.', 'S.e.', 'z', 'P>|z|',
		           '[95% Conf. int.]')
		span = [1]*5 + [2]
		etype = ['string']*6
		output += p.write_row(entries, span, etype)
		output += p.write_row('-'*p.table_width, [1], ['string'])

		entries = p._reg_entries('Intercept', coef[0], se[0])
		span = [1]*7
		etype = ['string'] + ['float']*6
		output += p.write_row(entries, span, etype)

		lin = self._dict['lin']
		for i in xrange(len(lin)):
			entries = p._reg_entries('X'+str(lin[i]),
			                         coef[1+i], se[1+i])
			output += p.write_row(entries, span, etype)

		qua = self._dict['qua']
		for i in xrange(len(qua)):
			name = 'X'+str(qua[i][0])+'*X'+str(qua[i][1])
			entries = p._reg_entries(name, coef[1+len(lin)+i],
			                         se[1+len(lin)+i])
			output += p.write_row(entries, span, etype)

		return output


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


	def _compute_pscore(self, X):

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

		c, t = self._model.controls, self._model.treated
		X_c, X_t = X[c], X[t]
		N, K = X.shape

		neg_loglike = lambda x: self._neg_loglike(x, X_c, X_t)
		neg_gradient = lambda x: self._neg_gradient(x, X_c, X_t)

		logit = fmin_bfgs(neg_loglike, np.zeros(K), neg_gradient,
		                  full_output=True, disp=False)

		pscore = {}
		pscore['coef'], pscore['loglike'] = logit[0], -logit[1]
		pscore['fitted'] = np.empty(N)
		pscore['fitted'][c] = self._sigmoid(X_c.dot(pscore['coef']))
		pscore['fitted'][t] = self._sigmoid(X_t.dot(pscore['coef']))

		return pscore


	def _form_matrix(self, X, lin, qua):

		"""
		Forms covariate matrix for use in propensity score estimation,
		based on requirements on linear and quadratic terms.

		Expected args
		-------------
			X: matrix, ndarray
				Matrix of original covariates to form a matrix
				out of.
			lin: list
				Column numbers (zero-based) of the original
				covariate matrix to include linearly.
			qua: list
				Tuples indicating which columns of the original
				covariate matrix to multiply and include. E.g.,
				[(1,1), (2,3)] indicates squaring the 1st column
				and including the product of the 2nd and 3rd
				columns.

		Returns
		-------
			Covariate matrix formed based on requirements on linear
			and quadratic terms.
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


	def _compute_se(self):

		"""
		Computes standard errors for the coefficient estimates of the
		logistic regression used to estimate propensity scores.

		Returns
		-------
			Vector of standard errors, same dimension as vector
			of coefficient estimates.
		"""

		lin, qua = self._dict['lin'], self._dict['qua']
		mat = self._form_matrix(self._model.X, lin, qua)
 		p = self._dict['fitted']
		H = np.dot(p*(1-p)*mat.T,mat)
		
		return np.sqrt(np.diag(np.linalg.inv(H)))


class PropensitySelect(Propensity):

	"""
	Dictionary-like class containing propensity score data, including
	estimated logistic regression coefficients, predicted propensity score,
	maximized log-likelihood, and lists of the linear and quadratic terms
	that are included in the regression.
	"""

	def __init__(self, lin_B, C_lin, C_qua, model):

		self._model = model
		D, X = self._model.D, self._model.X

		if C_lin == np.inf:  # inf threshold, so include basic only
			lin = lin_B
		elif C_lin == 0:  # min threshold, so include everything
			lin = range(X.shape[1])
		else:
			pot = list(set(xrange(X.shape[1])) - set(lin_B))
			lin = self._select_terms(lin_B, pot, C_lin)

		if C_qua == np.inf:  # inf threshold, so include nothing
			qua = []
		elif C_qua == 0:  # min threshold, so include everything
			qua = list(combinations_with_replacement(lin, 2))
		else:
			pot = list(combinations_with_replacement(lin, 2))
			qua = self._select_terms([], pot, C_qua, lin)

		mat = self._form_matrix(X, lin, qua)
		self._dict = self._compute_pscore(mat)
		self._dict['lin'], self._dict['qua'] = lin, qua
		self._dict['se'] = None
		

	def _select_terms(self, cur, pot, crit, lin=[]):
	
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
			
		D, X = self._model.D, self._model.X

		# calculate log-likelihood under null of no additional terms
		if not lin:  # lin is empty, so linear terms not yet decided
			mat = self._form_matrix(X, cur, [])
			ll_null = self._compute_pscore(mat)['loglike']
		else:  # lin is not empty, so linear terms are already fixed
			mat = self._form_matrix(X, lin, cur)
			ll_null = self._compute_pscore(mat)['loglike']

		# calculate LR stat after including each additional term
		lr = np.empty(len(pot))
		if not lin:
			for i in xrange(len(pot)):
				mat = self._form_matrix(X, cur+[pot[i]], [])
				ll = self._compute_pscore(mat)['loglike']
				lr[i] = 2*(ll - ll_null)
		else:
			for i in xrange(len(pot)):
				mat = self._form_matrix(X, lin, cur+[pot[i]])
				ll = self._compute_pscore(mat)['loglike']
				lr[i] = 2*(ll - ll_null)

		argmax = np.argmax(lr)
		if lr[argmax] < crit:
			return cur  # stop including additional terms
		else:
			# include new term and recurse on remaining
			new_term = pot.pop(argmax)
			return self._select_terms(cur+[new_term],
			                          pot, crit, lin)

