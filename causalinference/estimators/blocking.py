from __future__ import division
import numpy as np

from base import Estimator


class Blocking(Estimator):

	"""
	Dictionary-like class containing treatment effect estimates. Standard
	errors are only computed when needed.
	"""

	def __init__(self, model):

		self._model = model
		super(Blocking, self).__init__()


	def _compute_est(self):

		"""
		Computes average treatment effects as a weighted average
		of within-bin regression estimates. Sample must be stratified
		first.

		Returns
		-------
			3-tuple of ATE, ATT, and ATC estimates, respectively.
		"""

		model = self._model
		N, N_c, N_t = model.N, model.N_c, model.N_t

		ate = np.sum([s.N/N*s.within for s in model.strata])
		att = np.sum([s.N_t/N_t*s.within for s in model.strata])
		atc = np.sum([s.N_c/N_c*s.within for s in model.strata])

		return (ate, att, atc)


	def _compute_se(self):

		"""
		Computes standard errors for average treatment effects
		estimated via regression within blocks.

		Returns
		-------
			3-tuple of ATE, ATT, and ATC standard error estimates,
			respectively.

		"""

		model = self._model
		N, N_c, N_t = model.N, model.N_c, model.N_t

		wvar = [(s.N/N)**2 * s.se**2 for s in model.strata] 
		wvar_t = [(s.N_t/N_t)**2 * s.se**2 for s in model.strata]
		wvar_c = [(s.N_c/N_c)**2 * s.se**2 for s in model.strata]
		ate_se = np.sqrt(np.array(wvar).sum())
		att_se = np.sqrt(np.array(wvar_t).sum())
		atc_se = np.sqrt(np.array(wvar_c).sum())

		return (ate_se, att_se, atc_se)

