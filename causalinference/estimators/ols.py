from __future__ import division
import numpy as np
import scipy.linalg

from .base import Estimator


class OLS(Estimator):

	"""
	Dictionary-like class containing treatment effect estimates.
	"""

	def __init__(self, data, adj):

		self._method = 'OLS'
		Y, D, X = data['Y'], data['D'], data['X']
		X_c, X_t = data['X_c'], data['X_t']

		Z = form_matrix(D, X, adj)
		olscoef = np.linalg.lstsq(Z, Y)[0]
		u = Y - Z.dot(olscoef)
		cov = calc_cov(Z, u)

		self._dict = dict()
		self._dict['ate'] = calc_ate(olscoef)
		self._dict['ate_se'] = calc_ate_se(cov)

		if adj == 2:
			Xmean = X.mean(0)
			meandiff_c = X_c.mean(0) - Xmean
			meandiff_t = X_t.mean(0) - Xmean
			self._dict['atc'] = calc_atx(olscoef, meandiff_c)
			self._dict['att'] = calc_atx(olscoef, meandiff_t)
			self._dict['atc_se'] = calc_atx_se(cov, meandiff_c)
			self._dict['att_se'] = calc_atx_se(cov, meandiff_t)


def form_matrix(D, X, adj):

	N, K = X.shape

	if adj == 0:
		cols = 2
	elif adj == 1:
		cols = 2+K
	else:
		cols = 2+2*K
	
	Z = np.empty((N, cols))
	Z[:, 0] = 1  # intercept term
	Z[:, 1] = D
	if adj >= 1:
		dX = X - X.mean(0)
		Z[:, 2:2+K] = dX
	if adj == 2:
		Z[:, 2+K:] = D[:, None] * dX

	return Z


def calc_ate(olscoef):

	return olscoef[1]  # coef of treatment variable


def calc_atx(olscoef, meandiff):

	K = (len(olscoef)-2) // 2

	return olscoef[1] + np.dot(meandiff, olscoef[2+K:])


def calc_cov(Z, u):

	A = np.linalg.inv(np.dot(Z.T, Z))
	B = np.dot(u[:, None]*Z, A)

	return np.dot(B.T, B)


def submatrix(cov):

	K = (cov.shape[0]-2) // 2
	submat = np.empty((1+K, 1+K))
	submat[0,0] = cov[1,1]
	submat[0,1:] = cov[1,2+K:]
	submat[1:,0] = cov[2+K:,1]
	submat[1:,1:] = cov[2+K:, 2+K:]

	return submat


def calc_ate_se(cov):

	return np.sqrt(cov[1,1])


def calc_atx_se(cov, meandiff):

	a = np.concatenate((np.array([1]), meandiff))

	return np.sqrt(a.dot(submatrix(cov)).dot(a))

