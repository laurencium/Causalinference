from __future__ import division
import numpy as np
from scipy.optimize import fmin_bfgs
from itertools import combinations_with_replacement

import causalinference.utils.tools as tools
from .data import Dict


class Propensity(Dict):

	"""
	Dictionary-like class containing propensity score data.
	
	Propensity score related data includes estimated logistic regression
	coefficients, maximized log-likelihood, predicted propensity scores,
	and lists of the linear and quadratic terms that are included in the
	logistic regression.
	"""

	def __init__(self, data, lin, qua):

		Z = form_matrix(data['X'], lin, qua)
		Z_c, Z_t = Z[data['controls']], Z[data['treated']]
		beta = calc_coef(Z_c, Z_t)

		self._data = data
		self._dict = dict()
		self._dict['lin'], self._dict['qua'] = lin, qua
		self._dict['coef'] = beta
		self._dict['loglike'] = -neg_loglike(beta, Z_c, Z_t)
		self._dict['fitted'] = sigmoid(Z.dot(beta))
		self._dict['se'] = calc_se(Z, self._dict['fitted'])


	def __str__(self):

		table_width = 80

		coefs = self._dict['coef']
		ses = self._dict['se']

		output = '\n'
		output += 'Estimated Parameters of Propensity Score\n\n'

		entries1 = ['', 'Coef.', 'S.e.', 'z', 'P>|z|',
		           '[95% Conf. int.]']
		entry_types1 = ['string']*6
		col_spans1 = [1]*5 + [2]
		output += tools.add_row(entries1, entry_types1,
		                        col_spans1, table_width)
		output += tools.add_line(table_width)

		entries2 = tools.gen_reg_entries('Intercept', coefs[0], ses[0])
		entry_types2 = ['string'] + ['float']*6
		col_spans2 = [1]*7
		output += tools.add_row(entries2, entry_types2,
		                        col_spans2, table_width)

		lin = self._dict['lin']
		for (lin_term, coef, se) in zip(lin, coefs[1:], ses[1:]):
			entries3 = tools.gen_reg_entries('X'+str(lin_term),
			                                 coef, se)
			output += tools.add_row(entries3, entry_types2,
			                        col_spans2, table_width)

		qua = self._dict['qua']
		lin_num = len(lin)+1  # including intercept
		for (qua_term, coef, se) in zip(qua, coefs[lin_num:],
		                                ses[lin_num:]):
			name = 'X'+str(qua_term[0])+'*X'+str(qua_term[1])
			entries4 = tools.gen_reg_entries(name, coef, se)
			output += tools.add_row(entries4, entry_types2,
			                        col_spans2, table_width)

		return output


class PropensitySelect(Propensity):

	"""
	Dictionary-like class containing propensity score data.
	
	Propensity score related data includes estimated logistic regression
	coefficients, maximized log-likelihood, predicted propensity scores,
	and lists of the linear and quadratic terms that are included in the
	logistic regression.
	"""

	def __init__(self, data, lin_B, C_lin, C_qua):

		X_c, X_t = data['X_c'], data['X_t']
		lin = select_lin_terms(X_c, X_t, lin_B, C_lin)
		qua = select_qua_terms(X_c, X_t, lin, C_qua)

		super(PropensitySelect, self).__init__(data, lin, qua)


def form_matrix(X, lin, qua):

	N, K = X.shape

	mat = np.empty((N, 1+len(lin)+len(qua)))
	mat[:, 0] = 1  # constant term

	current_col = 1
	if lin:
		mat[:, current_col:current_col+len(lin)] = X[:, lin]
		current_col += len(lin)
	for term in qua:  # qua is a list of tuples of column numbers
		mat[:, current_col] = X[:, term[0]] * X[:, term[1]]
		current_col += 1

	return mat


def sigmoid(x, top_threshold=100, bottom_threshold=-100):

	high_x = (x >= top_threshold)
	low_x = (x <= bottom_threshold)
	mid_x = ~(high_x | low_x)

	values = np.empty(x.shape[0])
	values[high_x] = 1.0
	values[low_x] = 0.0
	values[mid_x] = 1/(1+np.exp(-x[mid_x]))

	return values


def log1exp(x, top_threshold=100, bottom_threshold=-100):

	high_x = (x >= top_threshold)
	low_x = (x <= bottom_threshold)
	mid_x = ~(high_x | low_x)

	values = np.empty(x.shape[0])
	values[high_x] = 0.0
	values[low_x] = -x[low_x]
	values[mid_x] = np.log(1 + np.exp(-x[mid_x]))

	return values


def neg_loglike(beta, X_c, X_t):

	return log1exp(X_t.dot(beta)).sum() + log1exp(-X_c.dot(beta)).sum()


def neg_gradient(beta, X_c, X_t):

	return (sigmoid(X_c.dot(beta))*X_c.T).sum(1) - \
	       (sigmoid(-X_t.dot(beta))*X_t.T).sum(1)


def calc_coef(X_c, X_t):

	K = X_c.shape[1]

	neg_ll = lambda b: neg_loglike(b, X_c, X_t)
	neg_grad = lambda b: neg_gradient(b, X_c, X_t)

	logit = fmin_bfgs(neg_ll, np.zeros(K), neg_grad,
			  full_output=True, disp=False)

	return logit[0]


def calc_se(X, phat):

	H = np.dot(phat*(1-phat)*X.T, X)
	
	return np.sqrt(np.diag(np.linalg.inv(H)))


def get_excluded_lin(K, included):

	included_set = set(included)

	return [x for x in range(K) if x not in included_set]


def get_excluded_qua(lin, included):

	whole_set = list(combinations_with_replacement(lin, 2))
	included_set = set(included)

	return [x for x in whole_set if x not in included_set]


def calc_loglike(X_c, X_t, lin, qua):

	Z_c = form_matrix(X_c, lin, qua)
	Z_t = form_matrix(X_t, lin, qua)
	beta = calc_coef(Z_c, Z_t)

	return -neg_loglike(beta, Z_c, Z_t)


def select_lin(X_c, X_t, lin_B, C_lin):

	# Selects, through a sequence of likelihood ratio tests, the
	# variables that should be included linearly in propensity
	# score estimation.

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

	# Mostly a wrapper around function select_lin to handle cases that
	# require little computation.

	if C_lin <= 0:
		K = X_c.shape[1]
		return lin_B + get_excluded_lin(K, lin_B)
	elif C_lin == np.inf:
		return lin_B
	else:
		return select_lin(X_c, X_t, lin_B, C_lin)


def select_qua(X_c, X_t, lin, qua_B, C_qua):

	# Selects, through a sequence of likelihood ratio tests, the
	# variables that should be included quadratically in propensity
	# score estimation.

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

	# Mostly a wrapper around function select_qua to handle cases that
	# require little computation.

	if lin == []:
		return []
	if C_qua <= 0:
		return get_excluded_qua(lin, [])
	elif C_qua == np.inf:
		return []
	else:
		return select_qua(X_c, X_t, lin, [], C_qua)

