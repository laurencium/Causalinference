from __future__ import division
import numpy as np

from ..core import Dict


class Blocking(Dict):

	"""
	Dictionary-like class containing treatment effect estimates.
	"""

	def __init__(self, strata):
	
		self._dict = dict()
		for s in strata:
			s.est_via_ols()

		Ns = [s.raw_data['N'] for s in strata]
		N_cs = [s.raw_data['N_c'] for s in strata]
		N_ts = [s.raw_data['N_t'] for s in strata]
		ates = [s.estimates['ols']['ate'] for s in strata]
		atcs = [s.estimates['ols']['atc'] for s in strata]
		atts = [s.estimates['ols']['att'] for s in strata]

		self._dict['ate'] = calc_atx(ates, Ns)
		self._dict['atc'] = calc_atx(atcs, N_cs)
		self._dict['att'] = calc_atx(atts, N_ts)

		ate_ses = [s.estimates['ols']['ate_se'] for s in strata]
		atc_ses = [s.estimates['ols']['atc_se'] for s in strata]
		att_ses = [s.estimates['ols']['att_se'] for s in strata]

		self._dict['ate_se'] = calc_atx_se(ate_ses, Ns)
		self._dict['atc_se'] = calc_atx_se(atc_ses, N_cs)
		self._dict['att_se'] = calc_atx_se(att_ses, N_ts)


def calc_atx(atxs, Ns):

	N = sum(Ns)

	return np.sum(np.array(atxs) * np.array(Ns)) / N


def calc_atx_se(atx_ses, Ns):

	N = sum(Ns)
	var = np.sum(np.array(atx_ses)**2 * np.array(Ns)**2) / N**2

	return np.sqrt(var)

