from __future__ import division
import numpy as np

from ..core import Dict


class Weighting(Dict):

	"""
	Dictionary-like class containing treatment effect estimates.
	"""

	def __init__(self, data):

		Y, D, X = data['Y'], data['D'], data['X']
		pscore = data['pscore']

		weights = calc_weights(pscore, D)
		Y_w, Z_w = weigh_data(Y, D, X, weights)

		wlscoef = np.linalg.lstsq(Z_w, Y_w)[0]

		self._dict = dict()
		self._dict['ate'] = calc_ate(wlscoef)
		self._dict['atc'] = self._dict['ate']
		self._dict['att'] = self._dict['ate']


def calc_weights(pscore, D):

	N = pscore.shape[0]
	weights = np.empty(N)
	weights[D==0] = 1/(1-pscore[D==0])
	weights[D==1] = 1/pscore[D==1]

	return weights


def weigh_data(Y, D, X, weights):

	N, K = X.shape

	Y_w = weights * Y

	Z_w = np.empty((N,K+2))
	Z_w[:,0] = weights
	Z_w[:,1] = weights * D
	Z_w[:,2:] = weights[:,None] * X

	return (Y_w, Z_w)


def calc_ate(wlscoef):

	return wlscoef[1]  # coef of treatment variable

