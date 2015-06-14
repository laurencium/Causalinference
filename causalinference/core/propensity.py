from __future__ import division
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
		X = self._form_matrix(lin, qua)
		X_c, X_t = X[self._model.controls], X[self._model.treated]

		beta = self._calc_coef(X_c, X_t)

		self._dict = dict()
		self._dict['lin'], self._dict['qua'] = lin, qua
		self._dict['coef'] = beta
		self._dict['loglike'] = -self._neg_loglike(beta, X_c, X_t)
		self._dict['fitted'] = self._sigmoid(X.dot(beta))
		self._dict['se'] = None  # only compute on request


	def _form_matrix(self, lin, qua):

		"""
		Forms covariate matrix for use in propensity score estimation,
		based on requirements on linear and quadratic terms.

		Expected args
		-------------
			lin: string, list
				Column numbers (zero-based) of the original
				covariate matrix X to include linearly. Can
				alternatively be a string equal to 'all', which
				results in using whole covariate matrix.
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

		X, N, K = self._model.X, self._model.N, self._model.K

		if lin == 'all':
			mat = np.empty((N, 1+K+len(qua)))
		else:
			mat = np.empty((N, 1+len(lin)+len(qua)))
		mat[:, 0] = 1  # constant term

		current_col = 1
		if lin == 'all':
			mat[:, current_col:current_col+K] = X
			current_col += K
		elif lin:
			mat[:, current_col:current_col+len(lin)] = X[:, lin]
			current_col += len(lin)
		for term in qua:
			mat[:, current_col] = X[:, term[0]] * X[:, term[1]]
			current_col += 1

		return mat


	def _sigmoid(self, x, top_threshold=100, bottom_threshold=-100):
	
		"""
		Computes 1/(1+exp(-x)) for input x, to be used in maximum
		likelihood estimation of propensity score.

		Expected args
		-------------
			x: array-like
				numpy array, should be shaped (n,).
			top_threshold: scalar
				cut-off for large x to avoid evaluting exp(-x).
			bottom_threshold:
				cut-off for small x to avoid evaluting exp(-x).

		Returns
		-------
			Vector of 1/(1+exp(-x)) values.
		"""

		high_x = (x >= top_threshold)
		low_x = (x <= bottom_threshold)
		mid_x = ~(high_x | low_x)

		values = np.empty(x.shape[0])
		values[high_x] = 1.0
		values[low_x] = 0.0
		values[mid_x] = 1/(1+np.exp(-x[mid_x]))

		return values


	def _log1exp(self, x, top_threshold=100, bottom_threshold=-100):

		"""
		Computes log(1+exp(-x)) for input x, to be used in maximum
		likelihood estimation of propensity score.

		Expected args
		-------------
			x: array-like
				numpy array, should be shaped (n,).
			top_threshold: scalar
				cut-off for large x to avoid evaluting exp(-x).
			bottom_threshold:
				cut-off for small x to avoid evaluting exp(-x).

		Returns
		-------
			Vector of log(1+exp(-x)) values.
		"""

		high_x = (x >= top_threshold)
		low_x = (x <= bottom_threshold)
		mid_x = ~(high_x | low_x)

		values = np.empty(x.shape[0])
		values[high_x] = 0.0
		values[low_x] = -x[low_x]
		values[mid_x] = np.log(1 + np.exp(-x[mid_x]))

		return values


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


	def _calc_coef(self, X_c, X_t):

		"""
		Estimates propensity score model via logistic regression. Uses
		BFGS algorithm for optimization.

		Expected args
		-------------
			X_c: matrix, ndarray
				Covariate matrix of the control units.
			X_t: matrix, ndarray
				Covariate matrix of the treated units.

		Returns
		-------
			Estimated logistic regression coefficients.
		"""

		K = X_c.shape[1]

		neg_loglike = lambda b: self._neg_loglike(b, X_c, X_t)
		neg_gradient = lambda b: self._neg_gradient(b, X_c, X_t)

		logit = fmin_bfgs(neg_loglike, np.zeros(K), neg_gradient,
		                  full_output=True, disp=False)

		return logit[0]  # coefficient estimates


	def _calc_se(self, X, p):

		"""
		Computes standard errors for the coefficient estimates of a
		logistic regression, given matrix of independent variables
		and fitted values from the regression.

		Expected args
		-------------
			X: matrix, ndarray
				Matrix of independent variables used in the
				logistic regression. If a constant term was
				included, this matrix should include a column
				of ones.
			p: array-like
				Vector of fitted values from the logistic
				regression.

		Returns
		-------
			Vector of standard errors, same dimension as vector
			of coefficient estimates.
		"""

		H = np.dot(p*(1-p)*X.T, X)
		
		return np.sqrt(np.diag(np.linalg.inv(H)))


	def _store_se(self):

		lin, qua = self._dict['lin'], self._dict['qua']
		X = self._form_matrix(lin, qua)
		p = self._dict['fitted']
		self._dict['se'] = self._calc_se(X, p)


	def __getitem__(self, key):

		if key == 'se' and self._dict['se'] is None:
			self._store_se()

		return self._dict[key]


	def __setitem__(self, key, value):

		if key == 'fitted':
			self._dict[key] = value
		else:
			raise TypeError("'" + self.__class__.__name__ +
			                "' object does not support item " +
					"assignment")


	def __iter__(self):

		return iter(self._dict)


	def __str__(self):

		if self._dict['se'] is None:
			self._store_se()

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

		if self._dict['lin'] == 'all':
			lin = range(self._model.X.shape[1])
		else:
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


class PropensitySelect(Propensity):


	def __init__(self, lin_B, C_lin, C_qua, model):

		self._model = model


	def _get_excluded_lin(self, included):

		K = self._model.X.shape[1]
		included_set = set(included)

		return [term for term in xrange(K) if term not in included_set]


	def _get_excluded_qua(self, lin, included):

		whole_set = list(combinations_with_replacement(lin, 2))
		included_set = set(included)

		return [term for term in whole_set if term not in included_set]


	def _calc_loglike(self, lin, qua):

		X = self._form_matrix(lin, qua)
		X_c, X_t = X[self._model.controls], X[self._model.treated]
		beta = self._calc_coef(X_c, X_t)

		return -self._neg_loglike(beta, X_c, X_t)


	def _test_lin(self, lin_B, C_lin):

		excl = self._get_excluded_lin(lin_B)
		if excl == []:
			return lin_B

		ll_null = self._calc_loglike(lin_B, [])

		def lr_stat_lin(lin_term):
			ll_alt = self._calc_loglike(lin_B+[lin_term], [])
			return 2 * (ll_alt - ll_null)

		lr_stats = np.array(map(lr_stat_lin, excl))
		argmax_lr = lr_stats.argmax()

		if lr_stats[argmax_lr] < C_lin:
			return lin_B
		else:
			return self._test_lin(lin_B+[excl[argmax_lr]], C_lin)


	def _select_lin_terms(self, lin_B, C_lin):

		if C_lin <= 0:
			return lin_B + self._get_excluded_lin(lin_B)
		elif C_lin == np.inf:
			return lin_B
		else:
			return self._test_lin(lin_B, C_lin)


	def _pick_qua(self, lin, qua_B, C_qua):

		excluded = self._get_excluded_qua(lin, qua_B)
		if excluded == []:
			return qua_B

		ll_null = self._calc_loglike(lin, qua_B)

		def lr_stat_qua(qua_term):
			ll_alt = self._calc_loglike(lin, qua_B+[qua_term])
			return 2 * (ll_alt - ll_null)

		lr_stats = map(lr_stat_qua, excluded)
		max_lr, argmax_lr = lr_stats.max(), lr_stats.argmax()

		if max_lr < C_qua:
			return qua_B
		else:
			self._pick_qua(lin, qua_B+[excluded[argmax_lr]], C_qua)


'''
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
		self._dict = self._calc_coef(mat)
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
			ll_null = self._calc_coef(mat)['loglike']
		else:  # lin is not empty, so linear terms are already fixed
			mat = self._form_matrix(X, lin, cur)
			ll_null = self._calc_coef(mat)['loglike']

		# calculate LR stat after including each additional term
		lr = np.empty(len(pot))
		if not lin:
			for i in xrange(len(pot)):
				mat = self._form_matrix(X, cur+[pot[i]], [])
				ll = self._calc_coef(mat)['loglike']
				lr[i] = 2*(ll - ll_null)
		else:
			for i in xrange(len(pot)):
				mat = self._form_matrix(X, lin, cur+[pot[i]])
				ll = self._calc_coef(mat)['loglike']
				lr[i] = 2*(ll - ll_null)

		argmax = np.argmax(lr)
		if lr[argmax] < crit:
			return cur  # stop including additional terms
		else:
			# include new term and recurse on remaining
			new_term = pot.pop(argmax)
			return self._select_terms(cur+[new_term],
			                          pot, crit, lin)
'''

