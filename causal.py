
import numpy as np
import random
import itertools
import statsmodels.api as sm
from scipy.stats import norm


class CausalModel(object):


	def __init__(self, Y, D, X):

		self.Y, self.D, self.X = Y, D, X
		self.N, self.k = self.X.shape
		self.treated, self.controls = np.nonzero(D)[0], np.nonzero(D==0)[0]
		self.Y_t, self.Y_c = self.Y[self.treated], self.Y[self.controls]
		self.X_t, self.X_c = self.X[self.treated], self.X[self.controls]
		self.N_t = np.count_nonzero(D)
		self.N_c = self.N - self.N_t


	def __polymatrix(self, X, poly):

		"""
		Construct matrix of polynomial terms to be used in synthetic control
		matching.

		Arguments
		---------
			X: array-like
				k Covariate values/vectors from which higher order polynomial
				terms are to be constructed.
			poly: Integer
				Degree of polynomial to generate.

		Returns
		-------
			Original covariate matrix appended with higher polynomial terms.
		"""

		terms = []
		for power in xrange(2, poly+1):
			terms.extend(list(itertools.combinations_with_replacement(range(self.k),
			                                                          power)))
		num_of_terms = len(terms)
		X_poly = np.ones((X.shape[0], num_of_terms))
		for i in xrange(num_of_terms):
			for j in terms[i]:
				X_poly[:, i] = X_poly[:, i] * X[:, j]
		return np.column_stack((X, X_poly))


	def __synthetic(self, X, X_m, Y, Y_m):

		"""
		Compute individual-level treatment effect by applying synthetic
		control method on input covariate matrix X. Automatically adds the
		constraint that the weights have to sum to 1. The public function
		self.synthetic is a wrapper around this.

		Arguments
		---------
			X: array-like
				Covariate matrix of units to find matches for.
			X_m: array-like
				Covariate matrix of units used to construct controls.
			Y: array-like
				Vector of outcomes of units to find matches for.
			Y_m: array-like
				Vector of outcomes of units used to construct controls.

		Returns
		-------
			Vector of individual-level treatment effects.
		"""

		N = X.shape[0]
		N_m = X_m.shape[0]

		X_m1 = np.row_stack((X_m.T, np.ones(N_m)))  # add row of 1's

		ITT = np.empty(N)

		for i in xrange(N):
			x1 = np.append(X[i], 1)  # append 1 to restrict weights to sum to 1
			w = np.linalg.lstsq(X_m1, x1)[0]
			ITT[i] = Y[i] - np.dot(w, Y_m)

		return ITT


	def synthetic(self, poly=0):

		"""
		Estimate the average treatment effects via synthetic controls.

		Terms of higher order polynomials of the covariates can be used in
		constructing synthetic controls.

		Arguments
		---------
			poly: integer
				Highest polynomial degree to match up to in constructing
				synthetic controls.

		Returns
		-------
			A Results class instance.
		"""
 
		ITT = np.empty(self.N)

		if poly > 1:
			X_t = self.__polymatrix(self.X_t, poly)
			X_c = self.__polymatrix(self.X_c, poly)
			ITT[self.treated] = self.__synthetic(X_t, X_c, self.Y_t, self.Y_c)
			ITT[self.controls] = -self.__synthetic(X_c, X_t, self.Y_c, self.Y_t)
		else:
			ITT[self.treated] = self.__synthetic(self.X_t, self.X_c, self.Y_t, self.Y_c)
			ITT[self.controls] = -self.__synthetic(self.X_c, self.X_t, self.Y_c, self.Y_t)

		return Results(ITT[self.treated].mean(), ITT.mean(), ITT[self.controls].mean())

	def __norm(self, dX, W):

		"""
		Calculate vector of norms given weighting matrix W.

		Arguments
		---------
			dX: array-like
				Matrix of covariate differences.
			W: array-like
				k-by-k weighting matrix or None for inverse variances.

		Returns
		-------
			Vector of distance measures.
		"""

		if W is None:
			return (dX**2 / self.Xvar).sum(axis=1)
		else:
			return (dX.dot(W)*dX).sum(axis=1)


	def __msmallest_with_ties(self, x, m):

		"""
		Finds indices of the m smallest entries in an array. Ties are
		included, so the number of indices can be greater than m. Algorithm
		is of order O(nt), where t is number of ties.

		Arguments
		---------
			x: array-like
				Array of numbers to find m smallest entries for.
			m: integer
				Number of smallest entries to find.

		Returns
		-------
			List of indices of smallest entries.
		"""

		par_indx = np.argpartition(x, m)  # partition around (m+1)th order stat
		
		if x[par_indx[:m]].max() < x[par_indx[m]]:  # mth < (m+1)th entry
			return list(par_indx[:m])
		elif x[par_indx[m]] < x[par_indx[m+1:]].min():  # (m+1)th < (m+2)th
			return list(par_indx[:m+1])
		else:  # mth = (m+1)th = (m+2)th, so increment and recurse
			return self.__msmallest_with_ties(x, m+2)


	def __matchmaking(self, X, X_m, m=1, W=None):

		"""
		Perform nearest-neigborhood matching using specified weighting
		matrix in measuring distance. Ties are included, so the number
		of matches for a given unit can be greater than m.

		Arguments
		---------
			X: array-like
				Observations to find matches for.
			X_m: array-like
				Pool of potential matches.
			m: integer
				Number of matches to find per unit.
			W: array-like
				k-by-k weighting matrix (or None for inverse variances)

		Returns
		-------
			List of matched indices.
		"""

		m_indx = []

		for i in xrange(X.shape[0]):
			m_indx.append(self.__msmallest_with_ties(self.__norm(X_m - X[i], W), m))

		return m_indx


	def __bias(self, m_indx, Y_m, X_m, X):

		"""
		Estimate bias resulting from imperfect matches using least squares.
		When estimating ATET, regression should use control units. When
		estimating ATUT, regression should use treated units.

		Arguments
		---------
			m_indx: list
				Index of indices of matched units.
			Y_m: array-like
				Vector of outcomes to regress.
			X_m: array-like
				Covariate matrix to regress on.
			X: array-like
				Covariate matrix of subjects under study.

		Returns
		-------
			Vector of estimated biases.
		"""

		flat_indx = list(itertools.chain.from_iterable(m_indx))

		X_m1 = np.column_stack((np.ones(len(flat_indx)), X_m[flat_indx]))
		b = np.linalg.lstsq(X_m1, Y_m[flat_indx])[0][1:]  # includes intercept

		N = X.shape[0]
		bias = np.empty(N)
		for i in xrange(N):
			bias[i] = np.dot(X[i] - X_m[m_indx[i]].mean(0), b)

		return bias


	def matching(self, wmatrix=None, matches=1, correct_bias=False):

		"""
		Estimate average treatment effects using matching with replacement.

		By default, the weighting matrix used in measuring distance is the
		inverse variance matrix. The Mahalanobis metric or other arbitrary
		weighting matrices can also be used instead.

		The number of matches per subject can also be specified. Ties entries
		are included, so the number of matches can be greater than specified
		for some subjects.

		Bias correction can optionally be done. For treated units, the bias
		resulting from imperfect matches is estimated by
			(X_t - X_c[matched]) * b,
		where b is the estimated coefficient from regressiong Y_c[matched] on
		X_c[matched]. For control units, the analogous procedure is used.
		For details, see Imbens and Rubin.

		Arguments
		---------
			wmatrix: string, array-like
				Distance measure to be used; acceptable values are None
				(Euclidean norm, default), string 'maha' for Mahalanobis
				metric, or any arbitrary k-by-k matrix.
			matches: integer
				The number of units to match to a given subject. Defaults
				to 1.
			correct_bias: Boolean
				Correct bias resulting from imperfect matches or not; defaults
				to no correction.
			order_by_pscore: Boolean, optional
				Determines order of match-making when matching without
				replacement.

		Returns
		-------
			A Results class instance.
		"""

		if wmatrix is None:
			self.Xvar = self.X.var(0)  # store vector of covariate variances
		elif wmatrix == 'maha':
			wmatrix = np.linalg.inv(np.cov(self.X, rowvar=False))

		m_indx_c = self.__matchmaking(self.X_c, self.X_t, matches, wmatrix)
		m_indx_t = self.__matchmaking(self.X_t, self.X_c, matches, wmatrix)

		ITT = np.empty(self.N)
		for i in xrange(self.N_c):
			ITT[self.controls[i]] = self.Y_t[m_indx_c[i]].mean() - self.Y_c[i]
		for i in xrange(self.N_t):
			ITT[self.treated[i]] = self.Y_t[i] - self.Y_c[m_indx_t[i]].mean()

		if correct_bias:
			ITT[self.controls] += self.__bias(m_indx_c, self.Y_t, self.X_t, self.X_c)
			ITT[self.treated] -= self.__bias(m_indx_t, self.Y_c, self.X_c, self.X_t)

		return Results(ITT[self.treated].mean(), ITT.mean(), ITT[self.controls].mean())


	def matching_without_replacement(self, wmatrix=None, order_by_pscore=False, correct_bias=False):

		"""
		Estimate average treatment effects using matching without replacement.

		By default, the weighting matrix used in measuring distance is the
		inverse variance matrix. The Mahalanobis metric or other arbitrary
		weighting matrices can also be used instead.

		Bias correction can optionally be done. For treated units, the bias
		resulting from imperfect matches is estimated by
			(X_t - X_c[matched]) * b,
		where b is the estimated coefficient from regressiong Y_c[matched] on
		X_c[matched]. For control units, the analogous procedure is used.
		For details, see Imbens and Rubin.

		Matching without replacement is computed via a greedy algorithm, where
		the best match for a treated unit is chosen without regard to the rest
		of the sample. By default, treated units are matched in the order that
		they come in in the data. An option to match in the order of
		descending estimated propensity score is provided.

		Arguments
		---------
			wmatrix: string, array-like
				Distance measure to be used; acceptable values are None
				(Euclidean norm, default), string 'maha' for Mahalanobis
				metric, or any arbitrary k-by-k matrix.
			order_by_pscore: Boolean, optional
				Determines order of match-making when matching without
				replacement.
			correct_bias: Boolean
				Correct bias resulting from imperfect matches or not; defaults
				to no correction.

		Returns
		-------
			A Results class instance.
		"""

		m_indx = np.zeros((self.N_t, 1), dtype=np.int)

		if wmatrix is None:
			self.Xvar = self.X.var(0)
		elif wmatrix == 'maha':
			wmatrix = np.linalg.inv(np.cov(self.X, rowvar=False))

		if order_by_pscore:
			pscore = sm.Logit(self.D, self.X).fit(disp=False).predict()  # estimate by logit
			order = np.argsort(pscore[self.D==1])[::-1]  # descending pscore order
		else:
			order = xrange(self.N_t)
			
		unmatched = range(self.N_c)
		for i in order:
			dX = self.X_c[unmatched] - self.X_t[i]
			m_indx[i] = unmatched.pop(np.argmin(self.__norm(dX, wmatrix)))

		ITT = self.Y_t - self.Y_c[m_indx]

		if correct_bias:
			ITT -= self.__bias(list(m_indx), self.Y_c, self.X_c, self.X_t)

		return Results(ITT.mean())


	def ols(self):

		"""
		Estimate average treatment effects using least squares.

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

		return Results(ITT.mean(), reg.params[1])


class Results(object):

	def __init__(self, ATET, ATE=None, ATUT=None):
		self.ATET, self.ATE, self.ATUT = ATET, ATE, ATUT

	def summary(self):

		if self.ATE:
			print 'Estimated Average Treatment Effect:', self.ATE
		print 'Estimated Average Treatment Effect on the Treated:', self.ATET
		if self.ATUT:
			print 'Estimated Average Treatment Effect on the Untreated:', self.ATUT









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


#def UseLalonde():

import pandas as pd

lalonde = pd.read_csv('ldw_exper.csv')

covariate_list = ['age', 'educ', 'black', 'hisp', 'married',
                  're74', 're75', 'u74', 'u75']

# don't know how to not convert to array first
Y = np.array(lalonde['re78'])
D = np.array(lalonde['t'])
X = np.array(lalonde[covariate_list])

#W = np.diag(1/X.var(0))

causal = CausalModel(Y, D, X)
print causal.matching(matches=4).ATE
print causal.matching(matches=4).ATET
print causal.matching(matches=1).ATET
print causal.matching(matches=4, correct_bias=True).ATET


#UseLalonde()