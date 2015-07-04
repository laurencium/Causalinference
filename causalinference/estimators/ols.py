from __future__ import division
import numpy as np
import scipy.linalg

from ..core import Dict


class OLS(Dict):

	def __init__(self, data):

		Y, D, X = data['Y'], data['D'], data['X']
		Y_c, Y_t = data['Y_c'], data['Y_t']
		X_c, X_t = data['X_c'], data['X_t']
		N_c, N_t = data['N_c'], data['N_t']

		Xmean = X.mean(0)
		meandiff_c = X_c.mean(0) - Xmean
		meandiff_t = X_t.mean(0) - Xmean
		Z = form_matrix(D, X)
		olscoef = np.linalg.lstsq(Z, Y)[0]
		u = Y - Z.dot(olscoef)
		subcov = calc_subcov(Z, u)

		self._dict = dict()
		self._dict['ate'] = calc_ate(olscoef)
		self._dict['atc'] = calc_atx(olscoef, meandiff_c)
		self._dict['att'] = calc_atx(olscoef, meandiff_t)
		self._dict['ate_se'] = calc_ate_se(subcov)
		self._dict['atc_se'] = calc_atx_se(subcov, meandiff_c)
		self._dict['att_se'] = calc_atx_se(subcov, meandiff_t)


def form_matrix(D, X):

	N, K = X.shape
	dX = X - X.mean(0)

	Z = np.empty((N, 2+2*K))
	Z[:, 0] = 1
	Z[:, 1] = D
	Z[:, 2:2+K] = D[:, None] * dX
	Z[:, 2+K:] = dX

	return Z


def calc_ate(olscoef):

	return olscoef[1]


def calc_atx(olscoef, meandiff):

	K = (len(olscoef)-2) / 2

	return olscoef[1] + np.dot(meandiff, olscoef[2:2+K])


def calc_subcov(Z, u):

	K = (Z.shape[1]-2) / 2
	A = np.linalg.inv(np.dot(Z.T, Z))
	B = np.dot(u[:,None]*Z, A[:,1:2+K])  # select columns for D, D*dX from A

	return np.dot(B.T, B)


def calc_ate_se(subcov):

	return np.sqrt(subcov[0,0])


def calc_atx_se(subcov, meandiff):

	a = np.concatenate((np.array([1]), meandiff))

	return np.sqrt(a.dot(subcov).dot(a))

