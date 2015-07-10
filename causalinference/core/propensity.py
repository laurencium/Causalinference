from __future__ import division
import numpy as np
from scipy.optimize import fmin_bfgs
from itertools import combinations_with_replacement

from data import Dict
from ..utils.tools import Printer


class Propensity(Dict):

	"""
	Dictionary-like class containing propensity score data.
	
	Propensity score related data includes estimated logistic regression
	coefficients, maximized log-likelihood, predicted propensity scores,
	and lists of the linear and quadratic terms that are included in the
	logistic regression.
	"""

	def __init__(self, lin, qua, data):

		Z = form_matrix(data['X'], lin, qua)
		Z_c, Z_t = Z[data['controls']], Z[data['treated']]
		beta = calc_coef(Z_c, Z_t)

		self._data = data
		self._dict = dict()
		self._dict['lin'], self._dict['qua'] = lin, qua
		self._dict['coef'] = beta
		self._dict['loglike'] = -neg_loglike(beta, Z_c, Z_t)
		self._dict['fitted'] = sigmoid(Z.dot(beta))
		self._dict['se'] = None  # only compute on request


	def _calc_and_store_se(self):

		"""
		Computes and stores standard errors of coefficient
		estimates. Only invoked when client calls a function that
		requires using or displaying standard errors, as standard
		error computation can be expensive.
		"""

		Z = form_matrix(self._data['X'], self['lin'], self['qua'])
		self._dict['se'] = calc_se(Z, self['fitted'])


	def __getitem__(self, key):

		if key == 'se' and self._dict['se'] is None:
			self._calc_and_store_se()

		return self._dict[key]


	def __repr__(self):

		if self._dict['se'] is None:
			self._calc_and_store_se()

		return self._dict.__repr__()


	def __str__(self):

		if self._dict['se'] is None:
			self._calc_and_store_se()

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


class PropensitySelect(Propensity):

	"""
	Dictionary-like class containing propensity score data.
	
	Propensity score related data includes estimated logistic regression
	coefficients, maximized log-likelihood, predicted propensity scores,
	and lists of the linear and quadratic terms that are included in the
	logistic regression.
	"""

	def __init__(self, lin_B, C_lin, C_qua, data):

		X_c, X_t = data['X_c'], data['X_t']
		lin = select_lin_terms(X_c, X_t, lin_B, C_lin)
		qua = select_qua_terms(X_c, X_t, lin, C_qua)

		super(PropensitySelect, self).__init__(lin, qua, data)


def form_matrix(X, lin, qua):

	"""
	Forms covariate matrix for use in propensity score estimation,
	based on requirements on linear and quadratic terms.

	Expected args
	-------------
		X: matrix, ndarray
			Matrix from which columns are selected to form
			covariate matrix for propensity score estimation.
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

	N, K = X.shape

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


def sigmoid(x, top_threshold=100, bottom_threshold=-100):

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


def log1exp(x, top_threshold=100, bottom_threshold=-100):

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


def neg_loglike(beta, X_c, X_t):

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

	return log1exp(X_t.dot(beta)).sum() + log1exp(-X_c.dot(beta)).sum()


def neg_gradient(beta, X_c, X_t):

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

	return (sigmoid(X_c.dot(beta))*X_c.T).sum(1) - \
	       (sigmoid(-X_t.dot(beta))*X_t.T).sum(1)


def calc_coef(X_c, X_t):

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

	neg_ll = lambda b: neg_loglike(b, X_c, X_t)
	neg_grad = lambda b: neg_gradient(b, X_c, X_t)

	logit = fmin_bfgs(neg_ll, np.zeros(K), neg_grad,
			  full_output=True, disp=False)

	return logit[0]  # coefficient estimates


def calc_se(X, p):

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


def get_excluded_lin(K, included):

	"""
	Finds excluded linear terms given a list of included ones.

	Expected args
	-------------
		K: int
			Number of covariates, to infer all linear terms.
		included: list
			Column numbers (zero-based) of the original
			covariate matrix that have been included
			linearly.

	Returns
	-------
		List of column numbers of the original covariate matrix
		corresponding to the excluded linear terms.
	"""

	included_set = set(included)

	return [x for x in xrange(K) if x not in included_set]


def get_excluded_qua(lin, included):

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

	return [x for x in whole_set if x not in included_set]


def calc_loglike(X_c, X_t, lin, qua):

	"""
	Calculates log-likelihood given linear and quadratic terms to
	include in the logistic regression.

	Expected args
	-------------
		X_c: matrix, ndarray
			Original covariate matrix for the control units.
		X_t: matrix, ndarray
			Original covariate matrix for the treated units.
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

	Z_c = form_matrix(X_c, lin, qua)
	Z_t = form_matrix(X_t, lin, qua)
	beta = calc_coef(Z_c, Z_t)

	return -neg_loglike(beta, Z_c, Z_t)


def select_lin(X_c, X_t, lin_B, C_lin):

	"""
	Selects, through a sequence of likelihood ratio tests, the
	variables that should be included linearly in propensity
	score estimation. The covariate selection algorithm is
	described in Imbens and Rubin (2015).

	Expected args
	-------------
		X_c: matrix, ndarray
			Original covariate matrix for the control units.
		X_t: matrix, ndarray
			Original covariate matrix for the treated units.
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

	K = X_c.shape[1]
	excluded = get_excluded_lin(K, lin_B)
	if excluded == []:
		return lin_B

	ll_null = calc_loglike(X_c, X_t, lin_B, [])

	def lr_stat_lin(lin_term):
		ll_alt = calc_loglike(X_c, X_t, lin_B+[lin_term], [])
		return 2 * (ll_alt - ll_null)

	lr_stats = np.array([lr_stat_lin(term) for term in excluded])
	argmax_lr = lr_stats.argmax()

	if lr_stats[argmax_lr] < C_lin:
		return lin_B
	else:
		new_term = [excluded[argmax_lr]]
		return select_lin(X_c, X_t, lin_B+new_term, C_lin)


def select_lin_terms(X_c, X_t, lin_B, C_lin):

	"""
	Selects the variables that should be included linearly in
	propensity score estimation. Mostly a wrapper around function
	select_lin to handle cases that require little computation.
	
	Expected args
	-------------
		X_c: matrix, ndarray
			Original covariate matrix for the control units.
		X_t: matrix, ndarray
			Original covariate matrix for the treated units.
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
		K = X_c.shape[1]
		return lin_B + get_excluded_lin(K, lin_B)
	elif C_lin == np.inf:
		return lin_B
	else:
		return select_lin(X_c, X_t, lin_B, C_lin)


def select_qua(X_c, X_t, lin, qua_B, C_qua):

	"""
	Selects, through a sequence of likelihood ratio tests, the
	variables that should be included quadratically in propensity
	score estimation. The covariate selection algorithm is
	described in Imbens and Rubin (2015).

	Expected args
	-------------
		X_c: matrix, ndarray
			Original covariate matrix for the control units.
		X_t: matrix, ndarray
			Original covariate matrix for the treated units.
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
		the likelihood ratio tests.

	References
	----------
		Imbens, G. & Rubin, D. (2015). Causal Inference in
			Statistics, Social, and Biomedical Sciences: An
			Introduction.
	"""

	excluded = get_excluded_qua(lin, qua_B)
	if excluded == []:
		return qua_B

	ll_null = calc_loglike(X_c, X_t, lin, qua_B)

	def lr_stat_qua(qua_term):
		ll_alt = calc_loglike(X_c, X_t, lin, qua_B+[qua_term])
		return 2 * (ll_alt - ll_null)

	lr_stats = np.array([lr_stat_qua(term) for term in excluded])
	argmax_lr = lr_stats.argmax()

	if lr_stats[argmax_lr] < C_qua:
		return qua_B
	else:
		new_term = [excluded[argmax_lr]]
		return select_qua(X_c, X_t, lin, qua_B+new_term, C_qua)


def select_qua_terms(X_c, X_t, lin, C_qua):

	"""
	Selects the variables that should be included quadratically in
	propensity score estimation. Mostly a wrapper around function
	select_qua to handle cases that require little computation.
	
	Expected args
	-------------
		X_c: matrix, ndarray
			Original covariate matrix for the control units.
		X_t: matrix, ndarray
			Original covariate matrix for the treated units.
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
		return get_excluded_qua(lin, [])
	elif C_qua == np.inf:
		return []
	else:
		return select_qua(X_c, X_t, lin, [], C_qua)

