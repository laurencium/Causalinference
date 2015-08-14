from __future__ import division
import numpy as np
from itertools import chain

from base import Estimator


class Matching(Estimator):

	"""
	Dictionary-like class containing treatment effect estimates. Standard
	errors are only computed when needed.
	"""

	def __init__(self, data, W, m, bias_adj):

		self._method = 'Matching'
		N, N_c, N_t = data['N'], data['N_c'], data['N_t']
		Y_c, Y_t = data['Y_c'], data['Y_t']
		X_c, X_t = data['X_c'], data['X_t']

		matches_c = [match(X_i, X_t, W, m) for X_i in X_c]
		matches_t = [match(X_i, X_c, W, m) for X_i in X_t]
		Yhat_c = np.array([Y_t[idx].mean() for idx in matches_c])
		Yhat_t = np.array([Y_c[idx].mean() for idx in matches_t])
		ITT_c = Yhat_c - Y_c
		ITT_t = Y_t - Yhat_t

		if bias_adj:
			bias_coefs_c = bias_coefs(matches_c, Y_t, X_t)
			bias_coefs_t = bias_coefs(matches_t, Y_c, X_c)
			bias_c = bias(X_c, X_t, matches_c, bias_coefs_c)
			bias_t = bias(X_t, X_c, matches_t, bias_coefs_t)
			ITT_c = ITT_c - bias_c
			ITT_t = ITT_t + bias_t

		self._dict = dict()
		self._dict['atc'] = ITT_c.mean()
		self._dict['att'] = ITT_t.mean()
		self._dict['ate'] = (N_c/N)*self['atc'] + (N_t/N)*self['att']

		# placeholder variances below
		atc_var, att_var = ITT_c.var(), ITT_t.var()
		ate_var = (N_c/N)**2*atc_var + (N_t/N)**2*att_var
		self._dict['atc_se'] = np.sqrt(atc_var)
		self._dict['att_se'] = np.sqrt(att_var)
		self._dict['ate_se'] = np.sqrt(ate_var)


def norm(X_i, X_m, W):

	dX = X_m - X_i
	if W.ndim == 1:
		return (dX**2 * W).sum(1)
	else:
		return (dX.dot(W)*dX).sum(1)


def smallestm(d, m):

	# partition around (m+1)th order stat
	par_idx = np.argpartition(d, m)

	if d[par_idx[:m]].max() < d[par_idx[m]]:  # m < (m+1)th
		return par_idx[:m]
	elif d[par_idx[m]] < d[par_idx[m+1:]].min():  # m+1 < (m+2)th
		return par_idx[:m+1]
	else:  # mth = (m+1)th = (m+2)th, so increment and recurse
		return smallestm(d, m+2)


def match(X_i, X_m, W, m):

	d = norm(X_i, X_m, W)

	return smallestm(d, m)


def bias_coefs(matches, Y_m, X_m):

	flat_idx = reduce(lambda x,y: np.concatenate((x,y)), matches)
	N, K = flat_idx.shape[0], X_m.shape[1]

	Y = Y_m[flat_idx]
	X = np.empty((N, K+1))
	X[:, 0] = 1  # intercept term
	X[:, 1:] = X_m[flat_idx]

	return np.linalg.lstsq(X, Y)[0][1:]  # don't need intercept coef


def bias(X, X_m, matches, coefs):

	X_m_mean = [X_m[idx].mean(0) for idx in matches]
	bias_list = [(X_j-X_i).dot(coefs) for X_i,X_j in zip(X, X_m_mean)]

	return np.array(bias_list)

