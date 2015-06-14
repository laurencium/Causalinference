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
		lin_parsed = self._parse_lin_terms(lin)
		qua_parsed = self._parse_qua_terms(qua)
		X = self._form_matrix(lin_parsed, qua_parsed)
		X_c, X_t = X[self._model.controls], X[self._model.treated]

		beta = self._calc_coef(X_c, X_t)

		self._dict = dict()
		self._dict['lin'], self._dict['qua'] = lin_parsed, qua_parsed
		self._dict['coef'] = beta
		self._dict['loglike'] = -self._neg_loglike(beta, X_c, X_t)
		self._dict['fitted'] = self._sigmoid(X.dot(beta))
		self._dict['se'] = None  # only compute on request


	def _parse_lin_terms(self, lin):

		"""
		Converts, if necessary, specification of linear terms given in
		strings to list of column numbers of the original covariate
		matrix.

		Expected args
		-------------
			lin: string, list
				Strings, such as 'all', or list of column
				numbers, that specifies which covariates to
				include as linear terms.

		Returns
		-------
			List of column numbers of covariate matrix specifying
			which variables to include linearly.
		"""

		if lin == 'all':
			return range(self._model.K)
		else:
			return lin


	def _parse_qua_terms(self, qua):

		"""
		Converts, if necessary, specification of quadratic terms given
		in strings to list of tuples of column numbers of the original
		covariate matrix.

		Expected args
		-------------
			qua: string, list
				Strings, such as 'all', or list of paris of
				column numbers, that specifies which covariates
				to include as quadratic terms.

		Returns
		-------
			List of tuples of column numbers of covariate matrix
			specifying which terms to include quadratically.
		"""

		if qua == 'all':
			lin_terms = xrange(self._model.K)
			return list(combinations_with_replacement(lin_terms, 2))
		else:
			return qua


	def _form_matrix(self, lin, qua):

		"""
		Forms covariate matrix for use in propensity score estimation,
		based on requirements on linear and quadratic terms.

		Expected args
		-------------
			lin: list
				Column numbers (zero-based) of the original
				covariate matrix to include linearly.
			qua: list
				Tuples indicating which columns of the original
				covariate matrix to multiply and include. E.g.,
				[(1,1), (2,3)] indicates squaring the 2nd column
				and including the product of the 3rd and 4th
				columns.

		Returns
		-------
			Covariate matrix formed based on requirements on linear
			and quadratic terms.
		"""

		X, N, K = self._model.X, self._model.N, self._model.K

		mat = np.empty((N, 1+len(lin)+len(qua)))
		mat[:, 0] = 1  # constant term

		current_col = 1
		if lin:
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

		"""
		Computes and stores standard errors of coefficient
		estimates. Only invoked when client calls a function that
		requires using or displaying standard errors, as standard
		error computation can be expensive.
		"""

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


	def __repr__(self):

		if self._dict['se'] is None:
			self._store_se()

		return self._dict.__repr__()


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

	"""
	Dictionary-like class containing propensity score data, including
	estimated logistic regression coefficients, predicted propensity score,
	maximized log-likelihood, and lists of the linear and quadratic terms
	that are included in the regression.
	"""

	def __init__(self, lin_B, C_lin, C_qua, model):

		self._model = model
		lin_B_parsed = self._parse_lin_terms(lin_B)
		lin = self._select_lin_terms(lin_B_parsed, C_lin)
		qua = self._select_qua_terms(lin, C_qua)

		super(PropensitySelect, self).__init__(lin, qua, self._model)


	def _get_excluded_lin(self, included):

		"""
		Finds excluded linear terms given a list of included ones.

		Expected args
		-------------
			included: list
				Column numbers (zero-based) of the original
				covariate matrix that have been included
				linearly.

		Returns
		-------
			List of column numbers of the original covariate matrix
			corresponding to the excluded linear terms.
		"""

		K = self._model.X.shape[1]
		included_set = set(included)

		return [term for term in xrange(K) if term not in included_set]


	def _get_excluded_qua(self, lin, included):

		"""
		Finds excluded quadratic terms given a list of base linear
		terms and a list of included quadratic terms.

		Expected args
		-------------
			lin : list
				Column numbers (zero-based) of the original
				covariate matrix that have been included
				linearly. Quadratic terms considered are
				constructed out of these linear terms.
			included: list
				Tuples indicating pairs of columns of the
				original covariate matrix that have been
				included as quadratic terms. E.g.,
				[(1,1), (2,3)] indicates squaring the 2nd column
				and including the product of the 3rd and 4th
				columns.

		Returns
		-------
			List of tuples of column numbers of the original
			covariate matrix corresponding to the excluded
			quadratic terms.
		"""

		whole_set = list(combinations_with_replacement(lin, 2))
		included_set = set(included)

		return [term for term in whole_set if term not in included_set]


	def _calc_loglike(self, lin, qua):

		"""
		Calculates log-likelihood given linear and quadratic terms to
		include in the logistic regression.

		Expected args
		-------------
			lin : list
				Column numbers (zero-based) of the original
				covariate matrix to include.
			included: list
				Tuples indicating pairs of columns of the
				original covariate matrix to include. E.g.,
				[(1,1), (2,3)] indicates squaring the 2nd column
				and including the product of the 3rd and 4th
				columns.

		Returns
		-------
			Maximized log-likelihood value.
		"""

		X = self._form_matrix(lin, qua)
		X_c, X_t = X[self._model.controls], X[self._model.treated]
		beta = self._calc_coef(X_c, X_t)

		return -self._neg_loglike(beta, X_c, X_t)


	def _test_lin(self, lin_B, C_lin):

		"""
		Selects, through a sequence of likelihood ratio tests, the
		variables that should be included linearly in propensity
		score estimation. The covariate selection algorithm is
		described in Imbens and Rubin (2015).

		Expected args
		-------------
			lin_B: list
				Column numbers (zero-based) of the original
				covariate matrix that should be included as
				linear terms regardless.
			C_lin: scalar
				Critical value used in likelihood ratio test
				to decide whether candidate linear terms should
				be included.

		Returns
		-------
			List of column numbers of the original covariate matrix
			to include linearly as decided by the LR tests.

		References
		----------
			Imbens, G. & Rubin, D. (2015). Causal Inference in
				Statistics, Social, and Biomedical Sciences: An
				Introduction.
		"""

		excluded = self._get_excluded_lin(lin_B)
		if excluded == []:
			return lin_B

		ll_null = self._calc_loglike(lin_B, [])

		def lr_stat_lin(lin_term):
			ll_alt = self._calc_loglike(lin_B+[lin_term], [])
			return 2 * (ll_alt - ll_null)

		lr_stats = np.array(map(lr_stat_lin, excluded))
		argmax_lr = lr_stats.argmax()

		if lr_stats[argmax_lr] < C_lin:
			return lin_B
		else:
			new_term = [excluded[argmax_lr]]
			return self._test_lin(lin_B+new_term, C_lin)


	def _select_lin_terms(self, lin_B, C_lin):

		"""
		Selects the variables that should be included linearly in
		propensity score estimation. Mostly a wrapper around function
		_test_lin to handle cases that require little computation.
		
		Expected args
		-------------
			lin_B: list
				Column numbers (zero-based) of the original
				covariate matrix that should be included as
				linear terms regardless.
			C_lin: scalar
				Critical value used in likelihood ratio test
				to decide whether candidate linear terms should
				be included.

		Returns
		-------
			List of column numbers of the original covariate matrix
			selected to be included linearly.
		"""

		if C_lin <= 0:
			return lin_B + self._get_excluded_lin(lin_B)
		elif C_lin == np.inf:
			return lin_B
		else:
			return self._test_lin(lin_B, C_lin)


	def _test_qua(self, lin, qua_B, C_qua):

		"""
		Selects, through a sequence of likelihood ratio tests, the
		variables that should be included quadratically in propensity
		score estimation. The covariate selection algorithm is
		described in Imbens and Rubin (2015).

		Expected args
		-------------
			lin: list
				Column numbers (zero-based) of the original
				covariate matrix that have been included
				linearly. Quadratic terms considered are
				constructed out of these linear terms.
			qua_B: list
				Tuples of column numbers of the original
				covariate matrix that have already passed the
				test.
			C_qua: scalar
				Critical value used in likelihood ratio test
				to decide whether candidate quadratic terms
				should be included.

		Returns
		-------
			List of tuples of column numbers of the original
			covariate matrix to include quadratically as decided by
			the LR tests.

		References
		----------
			Imbens, G. & Rubin, D. (2015). Causal Inference in
				Statistics, Social, and Biomedical Sciences: An
				Introduction.
		"""

		excluded = self._get_excluded_qua(lin, qua_B)
		if excluded == []:
			return qua_B

		ll_null = self._calc_loglike(lin, qua_B)

		def lr_stat_qua(qua_term):
			ll_alt = self._calc_loglike(lin, qua_B+[qua_term])
			return 2 * (ll_alt - ll_null)

		lr_stats = np.array(map(lr_stat_qua, excluded))
		argmax_lr = lr_stats.argmax()

		if lr_stats[argmax_lr] < C_qua:
			return qua_B
		else:
			new_term = [excluded[argmax_lr]]
			return self._test_qua(lin, qua_B+new_term, C_qua)


	def _select_qua_terms(self, lin, C_qua):

		"""
		Selects the variables that should be included quadratically in
		propensity score estimation. Mostly a wrapper around function
		_test_qua to handle cases that require little computation.
		
		Expected args
		-------------
			lin: list
				Column numbers (zero-based) of the original
				covariate matrix that have been included
				linearly. Quadratic terms considered are
				constructed out of these linear terms.
			C_qua: scalar
				Critical value used in likelihood ratio test
				to decide whether candidate quadratic terms
				should be included.

		Returns
		-------
			List of tuples of column numbers of the original
			covariate matrix selected to be included quadratically.
		"""

		if lin == []:
			return []
		if C_qua <= 0:
			return self._get_excluded_qua(lin, [])
		elif C_qua == np.inf:
			return []
		else:
			return self._test_qua(lin, [], C_qua)

