
import numpy as np
import random
import itertools
import statsmodels.api as sm
from scipy.stats import norm


class CausalModel(object):

	def __init__(self, Y, D, X):

		self.Y, self.D, self.X = Y, D, X
		self.N, self.k = self.X.shape
		control, treated = (self.D==0), (self.D==1)
		self.Y_c, self.Y_t = self.Y[control], self.Y[treated]
		self.X_c, self.X_t = self.X[control], self.X[treated]
		self.N_c, self.N_t = self.X_c.shape[0], self.X_t.shape[0]


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
		return np.hstack((X, X_poly))


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

			X_control = self.__polymatrix(self.X_c, terms)
			
			for i in xrange(self.N_t):  # can vectorize, but higher space requirement
				X_treated = self.__polymatrix(self.X_t[i, :].reshape((1, self.k)), terms)
				w = np.linalg.lstsq(X_control.T, X_treated.flatten())[0]
				ITT[i] = self.Y_t[i] - np.dot(w, self.Y_c)

		else:
			for i in xrange(self.N_t):
				w = np.linalg.lstsq(self.X_c.T, self.X_t[i, ])[0]
				ITT[i] = self.Y_t[i] - np.dot(w, self.Y_c)

		return Results(self, ITT.mean(), ITT)


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


	def matching(self, replace=False, wmatrix=None, correct_bias=False,
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
				dX = self.X_c - self.X_t[i]  # N_c-by-k matrix of covariate differences
				match_index[i] = np.argmin(self.__norm(dX, wmatrix))
		else:
			unmatched = range(self.N_c)
			if order_by_pscore:
				pscore = sm.Logit(D, X).fit(disp=False).predict()  # estimate by logit
				order = np.argsort(pscore[D==1])[::-1]  # descending pscore order
			else:
				order = xrange(self.N_t)
			for i in order:
				dX = self.X_c[unmatched] - self.X_t[i]
				match_index[i] = unmatched.pop(np.argmin(self.__norm(dX, wmatrix)))

		if correct_bias:
			reg = sm.OLS(self.Y_c[match_index],
			             sm.add_constant(self.X_c[match_index])).fit()
			ITT = (self.Y_t - self.Y_c[match_index] - 
			       np.dot((self.X_t - self.X_c[match_index]), reg.params[1:]))
		else:
			ITT = self.Y_t - self.Y_c[match_index]

		return Results(self, ITT.mean(), ITT)


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

		return Results(self, ITT.mean(), ITT)


class Results(object):

	def __init__(self, model, ATT, ITT):
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
