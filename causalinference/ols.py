import numpy as np
import scipy.linalg


from .estimators import Estimator


class OLS(Estimator):

	def __init__(self, model):

		self._model = model
		super(OLS, self).__init__()


	def _compute_est(self):

		N, K = self._model.N, self._model.K
		Y, D = self._model.Y, self._model.D
		X, X_c, X_t = self._model.X, self._model.X_c, self._model.X_t

		Xmean = X.mean(0)
		self._Z = np.empty((N, 2+2*K))  # create design matrix
		self._Z[:,0] = 1  # constant term
		self._Z[:,1] = D
		self._Z[:,2:2+K] = D[:,None]*(X-Xmean)
		self._Z[:,-K:] = X

		Q, self._R = np.linalg.qr(self._Z)
		self._olscoef = scipy.linalg.solve_triangular(self._R,
		                                              Q.T.dot(Y))

		ate = self._olscoef[1]
		att = self._olscoef[1] + \
		      np.dot(X_t.mean(0)-Xmean, self._olscoef[2:2+K])
		atc = self._olscoef[1] + \
		      np.dot(X_c.mean(0)-Xmean, self._olscoef[2:2+K])

		return (ate, att, atc)


	def _compute_se(self):
	
		"""
		Computes standard errors for OLS estimates of ATE, ATT, and ATC.

		If Z denotes the design matrix (i.e., covariates, treatment
		indicator, product of the two, and a column of ones) and u
		denotes the vector of least squares residual, then the variance
		estimator can be found by computing White's heteroskedasticity-
		robust covariance matrix:
			inv(Z'Z) Z'diag(u^2)Z inv(Z'Z).
		The diagonal entry corresponding to the treatment indicator of
		this matrix is the appropriate variance estimate for ATE.
		Variance estimates for ATT and ATC are appropriately weighted
		sums of entries of the above matrix.
		"""

		N, K = self._model.N, self._model.K
		Y = self._model.Y
		X, X_c, X_t = self._model.X, self._model.X_c, self._model.X_t

		Xmean = X.mean(0)
		u = Y - self._Z.dot(self._olscoef)
		A = np.linalg.inv(np.dot(self._R.T, self._R))
		# select columns for D, D*dX from A
		B = np.dot(u[:,None]*self._Z, A[:,1:2+K])  
		covmat = np.dot(B.T, B)

		self._dict['ate_se'] = np.sqrt(covmat[0,0])
		C = np.empty(K+1); C[0] = 1
		C[1:] = X_t.mean(0)-Xmean
		self._dict['att_se'] = np.sqrt(C.dot(covmat).dot(C))
		C[1:] = X_c.mean(0)-Xmean
		self._dict['atc_se'] = np.sqrt(C.dot(covmat).dot(C))

