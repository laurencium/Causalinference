
import numpy as np
import random
import itertools
import statsmodels.api as sm
from scipy.stats import norm


class CausalModel(object):

	def __init__(self, Y, D, X):

		self.Y, self.D, self.X = Y, D, X
		self.N, self.k = self.X.shape
		self.treated = np.nonzero(D)
		self.controls = np.nonzero(D==0)
		self.N_t = np.count_nonzero(D)
		self.N_c = self.N - self.N_t


	def __polymatrix(self, X, terms):

		"""
		Constructs matrix of polynomial terms to be used in synthetic control matching.

		Arguments
		---------
			X: array-like
				k Covariate values/vectors from which higher order polynomial terms are
				to be constructed.
			terms: list
				List of specific combinations of covariates to multiply together.

		Returns
		-------
			Original covariates matrix appended with higher polynomial terms.
		"""

		num_of_terms = len(terms)
		X_poly = np.ones((X.shape[0], num_of_terms))
		for i in xrange(num_of_terms):
			for j in terms[i]:
				X_poly[:, i] = X_poly[:, i] * X[:, j]
		return np.column_stack((X, X_poly))


	def synthetic(self, poly=0):

		"""
		Estimates the average treatment effect for the treated (ATT) via synthetic
		controls.

		Terms of higher order polynomials of the covariates can be used in constructing
		synthetic controls.

		Arguments
		---------
			poly: integer
				Highest polynomial degree to match up to in constructing synthetic
				controls.

		Returns
		-------
			A Results class instance.
		"""

		ITT = np.zeros(self.N_t)

		if poly > 1:

			terms = []
			for power in xrange(2, poly+1):
				terms.extend(list(itertools.combinations_with_replacement(range(self.k),
				                                                          power)))
			num_of_terms = len(terms)

			X_control = self.__polymatrix(self.X[self.controls], terms)
			
			for i in xrange(self.N_t):  # can vectorize, but higher space requirement
				X_treated = self.__polymatrix(self.X[self.treated][i, :].reshape((1, self.k)), terms)
				w = np.linalg.lstsq(X_control.T, X_treated.ravel())[0]
				ITT[i] = self.Y[self.treated][i] - np.dot(w, self.Y[self.controls])

		else:
			for i in xrange(self.N_t):
				w = np.linalg.lstsq(self.X[self.controls].T, self.X[self.treated][i, ])[0]
				ITT[i] = self.Y[self.treated][i] - np.dot(w, self.Y[self.controls])

		return Results(ITT.mean(), ITT)


	def matching_old(self, replace=False, wmatrix=None, correct_bias=False,
	                 order_by_pscore=False):

		"""
		Estimates the average treatment effect for the treated (ATT) using matching.

		Can specify whether matching is done with or without replacement.

		By default, the distance metric used for measuring covariate differences
		is the Euclidean metric. The Mahalanobis metric or other arbitrary weighting
		matrices can also be used instead.

		Bias correction can optionally be done. Bias resulting from imperfect
		matches is estimated by
			(X_t - X_c[matched])' * b,
		where b is the estimated coefficient from regressiong Y_c[matched] on
		X_c[matched]. For details, see Imbens and Rubin.

		Matching without replacement is computed via a greedy algorithm, where
		the best match for a treated unit is chosen without regard to the rest of
		the sample. By default, treated units are matched in the order that they
		come in in the data. An option to match in the order of descending estimated
		propensity score is provided.

		Arguments
		---------
			replace: Boolean
				Match with replacement or not; defaults to without.
			wmatrix: string, array-like
				Distance measure to be used; acceptable values are None (Euclidean norm),
				string 'maha' for Mahalanobis metric, or any arbitrary k-by-k matrix
			correct_bias: Boolean
				Correct bias resulting from imperfect matches or not; defaults to
				no correction.
			order_by_pscore: Boolean, optional
				Determines order of match-making when matching without replacement.

		Returns
		-------
			A Results class instance.
		"""

		match_index = np.zeros(self.N_t, dtype=np.int)

		if wmatrix == 'maha':
			wmatrix = np.linalg.inv(np.cov(self.X, rowvar=False))

		if replace:
			for i in xrange(self.N_t):
				dX = self.X[self.controls] - self.X[self.treated][i]  # N_c-by-k matrix
				match_index[i] = np.argmin(self.__norm(dX, wmatrix))
		else:
			unmatched = range(self.N_c)
			if order_by_pscore:
				pscore = sm.Logit(self.D, self.X).fit(disp=False).predict()  # estimate by logit
				order = np.argsort(pscore[self.D==1])[::-1]  # descending pscore order
			else:
				order = xrange(self.N_t)
			for i in order:
				dX = self.X[self.controls][unmatched] - self.X[self.treated][i]
				match_index[i] = unmatched.pop(np.argmin(self.__norm(dX, wmatrix)))

		if correct_bias:
			reg = sm.OLS(self.Y[self.controls][match_index],
			             sm.add_constant(self.X[self.controls][match_index])).fit()
			ITT = (self.Y[self.treated] - self.Y[self.controls][match_index] - 
			       np.dot((self.X[self.treated] - self.X[self.controls][match_index]), reg.params[1:]))
		else:
			ITT = self.Y[self.treated] - self.Y[self.controls][match_index]

		return Results(ITT.mean(), ITT)


	def __norm(self, dX, W):

		"""
		Calculates a vector of norms given weighting matrix W.

		Arguments
		---------
			dX: array-like
				Matrix of covariate differences.
			W: array-like
				k-by-k weighting matrix or None for Eucliean norm.

		"""

		if W is None:
			return (dX**2).sum(axis=1)
		else:
			return (dX.dot(W)*dX).sum(axis=1)


	def __matchmaking(self, X, X_m, M=1, W=None):
		
		n = X.shape[0]
		m_indx = np.zeros((n, M), dtype=int)

		if M==1:
			for i in xrange(n):
				m_indx[i] = np.argmin(self.__norm(X_m - X[i], W))
		else:
			for i in xrange(n):
				m_indx[i] = np.argpartition(self.__norm(X_m - X[i], W), M)[:M]

		return m_indx


	def __bias(self, treated, m_indx):

		if treated:
			b = np.linalg.lstsq(np.column_stack((np.ones(m_indx.size), self.X[self.controls][m_indx.ravel()])),
			                    self.Y[self.controls][m_indx.ravel()])[0][1:]
			return np.dot(self.X[self.treated] - self.X[self.controls][m_indx].mean(1), b)
		else:
			b = np.linalg.lstsq(np.column_stack((np.ones(m_indx.size), self.X[self.treated][m_indx.ravel()])),
			                    self.Y[self.treated][m_indx.ravel()])[0][1:]
			return np.dot(self.X[self.treated][m_indx].mean(1) - self.X[self.controls], b)


	def matching(self, wmatrix=None, matches=1, correct_bias=False):

		if wmatrix == 'maha':
			wmatrix = np.linalg.inv(np.cov(self.X, rowvar=False))

		m_indx_c = self.__matchmaking(self.X[self.controls], self.X[self.treated], matches, wmatrix)
		m_indx_t = self.__matchmaking(self.X[self.treated], self.X[self.controls], matches, wmatrix)

		ITT = np.zeros(self.N)
		ITT[self.controls] = self.Y[self.treated][m_indx_c].mean(1)- self.Y[self.controls]
		ITT[self.treated] = self.Y[self.treated] - self.Y[self.controls][m_indx_t].mean(1)

		if correct_bias:
			ITT[self.controls] -= self.__bias(treated=False, m_indx=m_indx_c)
			ITT[self.treated] -= self.__bias(treated=True, m_indx=m_indx_t)

		return Results(ITT[self.treated].mean(), ITT)


	def matching_without_replacement(self, wmatrix=None, order_by_pscore=False, correct_bias=False):

		m_indx = np.zeros((self.N_t, 1), dtype=np.int)

		if wmatrix == 'maha':
			wmatrix = np.linalg.inv(np.cov(self.X, rowvar=False))

		if order_by_pscore:
			pscore = sm.Logit(self.D, self.X).fit(disp=False).predict()  # estimate by logit
			order = np.argsort(pscore[self.D==1])[::-1]  # descending pscore order
		else:
			order = xrange(self.N_t)
			
		unmatched = range(self.N_c)
		for i in order:
			dX = self.X[self.controls][unmatched] - self.X[self.treated][i]
			m_indx[i] = unmatched.pop(np.argmin(self.__norm(dX, wmatrix)))

		ITT = self.Y[self.treated] - self.Y[self.controls][m_indx]

		if correct_bias:
			ITT -= self.__bias(treated=True, m_indx=m_indx)

		return Results(ITT.mean(), ITT)


	def ols(self):

		"""
		Estimates the average treatment effect for the treated (ATT) using least squares.

		Returns
		-------
			A Results class instance.
		"""

		D = self.D.reshape((self.N, 1))
		dX = self.X - self.X.mean(0)
		DdX = D * dX
		Z = np.column_stack((D, dX, DdX))

		reg = sm.OLS(self.Y, sm.add_constant(Z)).fit()
		ITT = reg.params[1] + np.dot(dX[self.D==1], reg.params[-self.k:])

		return Results(ITT.mean(), ITT)


class Results(object):

	def __init__(self, ATT, ITT):
		self.ATT = ATT
		self.ITT = ITT

	def summary(self):

		print 'Estimated Average Treatment Effect on the Treated:', self.ATT









class parameters(object):

	"""
	Class object that stores the parameter values for use in the baseline model.
	
	See SimulateData function for model description.

	Args:
		N = Sample size (control + treated units) to generate
		k = Number of covariates
	"""

	def __init__(self, N=500, k=3):  # set initial parameter values
		self.N = N  # sample size (control + treated units)
		self.k = k  # number of covariates

		self.delta = 3
		self.beta = np.ones(k)
		self.theta = np.ones(k)
		self.mu = np.zeros(k)
		self.Sigma = np.identity(k)
		self.Gamma = np.identity(2)


def SimulateData(para=parameters(), nonlinear=False, return_counterfactual=False):

	"""
	Function that generates data according to one of two simple models that
	satisfies the Unconfoundedness assumption.

	The covariates and error terms are generated according to
		X ~ N(mu, Sigma), epsilon ~ N(0, Gamma).
	The counterfactual outcomes are generated by
		Y_0 = X*beta + epsilon_0,
		Y_1 = delta + X*(beta+theta) + epsilon_1,
	if the nonlinear Boolean is False, or by
		Y_0 = sum_of_abs(X) + epsilon_0,
		Y_1 = sum_of_squares(X) + epsilon_1,
	if the nonlinear Boolean is True.

	Selection is done according to the following propensity score function:
		P(D=1|X) = Phi(X*beta),
	where Phi is the standard normal CDF.

	Args:
		para = Model parameter values supplied by parameter class object
		nonlinear = Boolean indicating whether the data generating model should
		            be linear or not
		return_counterfactual = Boolean indicating whether to return vectors of
		                        counterfactual outcomes

	Returns:
		Y = N-dimensional array of observed outcomes
		D = N-dimensional array of treatment indicator; 1=treated, 0=control
		X = N-by-k matrix of covariates
		Y_0 = N-dimensional array of non-treated outcomes
		Y_1 = N-dimensional array of treated outcomes
	"""

	k = len(para.mu)

	X = np.random.multivariate_normal(mean=para.mu, cov=para.Sigma,
	                                  size=para.N)
	epsilon = np.random.multivariate_normal(mean=np.zeros(2), cov=para.Gamma,
	                                        size=para.N)

	Xbeta = np.dot(X, para.beta)

	pscore = norm.cdf(Xbeta)
	# for each p in pscore, generate Bernoulli rv with success probability p
	D = np.array([np.random.binomial(1, p, size=1) for p in pscore]).flatten()

	if nonlinear:
		Y_0 = abs(X).sum(1) + epsilon[:, 0]
		Y_1 = (X**2).sum(1) + epsilon[:, 1]
	else:
		Y_0 = Xbeta + epsilon[:, 0]
		Y_1 = para.delta + np.dot(X, para.beta+para.theta) + epsilon[:, 1]

	Y = (1 - D) * Y_0 + D * Y_1  # compute observed outcome

	if return_counterfactual:
		return Y, D, X, Y_0, Y_1
	else:
		return Y, D, X
