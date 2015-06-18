import numpy as np

from base import Estimator


class Weighting(Estimator):

	"""
	Dictionary-like class containing treatment effect estimates. Standard
	errors are only computed when needed.
	"""

	def __init__(self, model):

		self._model = model
		super(Weighting, self).__init__()


	def _ols_predict(self, Y, X, X_new):

		"""
		Estimates linear regression model with least squares and project
		based on new input data.

		Expected args
		-------------
			Y: array-like
				Vector of observed outcomes.
			X: matrix, ndarray
				Matrix of covariates to regress on.
			X_new: matrix, ndarray
				Matrix of covariates used to generate
				predictions.

		Returns
		-------
			Vector of predicted values.
		"""

		Z = np.empty((X.shape[0], X.shape[1]+1))
		Z[:, 0] = 1  # constant term
		Z[:, 1:] = X
		beta = np.linalg.lstsq(Z, Y)[0]

		return beta[0] + X_new.dot(beta[1:])


	def _compute_est(self):

		"""
		Computes treatment effects using the Horvitz-Thompson weighting
		estimator modified to incorporate covariates. Estimator
		possesses the so-called 'double robustness' property. See
		Lunceford and Davidian (2004) for details.

		Returns
		-------
			3-tuple of ATE, ATT, and ATC estimates, respectively.

		References
		----------
			Lunceford, J. K., & Davidian, M. (2004). Stratification
				and weighting via the propensity score in
				estimation of causal treatment effects: a
				comparative study. Statistics in Medicine.
		"""

		N, N_c, N_t = self._model.N, self._model.N_c, self._model.N_t
		Y, Y_c, Y_t = self._model.Y, self._model.Y_c, self._model.Y_t
		D = self._model.D
		c, t = self._model.controls, self._model.treated
		X, X_c, X_t = self._model.X, self._model.X_c, self._model.X_t
		phat = self._model.pscore['fitted']

		Yhat_t = self._ols_predict(Y_t, X_t, X)
		Yhat_c = self._ols_predict(Y_c, X_c, X)
		self._summand = (D-phat) * (Y - (1-phat)*Yhat_t - \
		                            phat*Yhat_c) / (phat*(1-phat))

		ate = self._summand.mean()
		att = self._summand[t].mean()
		atc = self._summand[c].mean()

		return (ate, att, atc)


	def _compute_se(self):

		"""
		Computes standard errors for weighting estimator. See Lunceford
		and Davidian (2004) for details.

		Returns
		-------
			3-tuple of ATE, ATT, and ATC standard error estimates,
			respectively.

		References
		----------
			Lunceford, J. K., & Davidian, M. (2004). Stratification
				and weighting via the propensity score in
				estimation of causal treatment effects: a
				comparative study. Statistics in Medicine.
		"""

		N, N_c, N_t = self._model.N, self._model.N_c, self._model.N_t
		c, t = self._model.controls, self._model.treated

		ate_se = np.sqrt(self._summand.var()/N)
		att_se = np.sqrt(self._summand[t].var()/N_t)
		atc_se = np.sqrt(self._summand[c].var()/N_c)

		return (ate_se, att_se, atc_se)

