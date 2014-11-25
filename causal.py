import numpy as np
import random
import itertools
import statsmodels.api as sm
from scipy.stats import norm
from scipy.optimize import fmin_bfgs
from math import sqrt
import cvxpy as cvx
from sklearn import linear_model


class CausalModel(object):


	def __init__(self, Y, D, X):

		self._Y, self._D, self._X = Y, D, X
		self._N, self._k = self._X.shape
		self._treated, self._controls = np.nonzero(D)[0], np.nonzero(D==0)[0]
		self._Y_t, self._Y_c = self._Y[self._treated], self._Y[self._controls]
		self._X_t, self._X_c = self._X[self._treated], self._X[self._controls]
		self._N_t = np.count_nonzero(D)
		self._N_c = self._N - self._N_t


	def normalized_diff(self):

		"""
		Computes normalized difference in covariates for assessing balance.

		Normalized difference is the difference in group means, scaled by the
		square root of the average of the two within-group variances. Large
		values indicate that simple linear adjustment methods may not be adequate
		for removing biases that are associated with differences in covariates.

		Unlike t-statistic, normalized differences do not, in expectation,
		increase with sample size, and thus is more appropriate for assessing
		balance.
		"""

		return (self._X_t.mean(0) - self._X_c.mean(0)) / np.sqrt((self._X_t.var(0) +
		                                                         self._X_c.var(0))/2)

	def _sigmoid(self, x):

		return 1/(1+np.exp(-x))


	def _logsigmoid(self, x):

		return np.log(self._sigmoid(x))


	def _loglike(self, beta, X_t, X_c):

		return self._logsigmoid(X_t.dot(beta)).sum() + \
		       self._logsigmoid(-X_c.dot(beta)).sum()


	def _gradient(self, beta, X_t, X_c):

		return (self._sigmoid(-X_t.dot(beta))*X_t.T).sum(1) - \
		       (self._sigmoid(X_c.dot(beta))*X_c.T).sum(1)


	def _pscore(self, X):

		X_t = X[self._D==1]
		X_c = X[self._D==0]
		K = X.shape[1]

		neg_loglike = lambda x: -self._loglike(x, X_t, X_c)
		neg_gradient = lambda x: -self._gradient(x, X_t, X_c)

		logit = fmin_bfgs(neg_loglike, np.zeros(K), neg_gradient, full_output=True)

		beta = logit[0]
		loglike = -logit[1]
		fitted = np.empty(self._N)
		fitted[self._D==1] = self._sigmoid(X_t.dot(beta))
		fitted[self._D==0] = self._sigmoid(X_c.dot(beta))

		return (beta, loglike, fitted)

		
	def _polymatrix(self, X, poly):

		"""
		Constructs matrix of polynomial terms to be used in synthetic control
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
			terms.extend(list(itertools.combinations_with_replacement(range(self._k),
			                                                          power)))
		num_of_terms = len(terms)
		X_poly = np.ones((X.shape[0], num_of_terms))
		for i in xrange(num_of_terms):
			for j in terms[i]:
				X_poly[:, i] = X_poly[:, i] * X[:, j]
		return np.column_stack((X, X_poly))


	def _synthetic(self, X, X_m, Y, Y_m):

		"""
		Computes individual-level treatment effect by applying synthetic
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
		Estimates the average treatment effects via synthetic controls.

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
 
		ITT = np.empty(self._N)

		if poly > 1:
			X_t = self._polymatrix(self._X_t, poly)
			X_c = self._polymatrix(self._X_c, poly)
			ITT[self._treated] = self._synthetic(X_t, X_c, self._Y_t, self._Y_c)
			ITT[self._controls] = -self._synthetic(X_c, X_t, self._Y_c, self._Y_t)
		else:
			ITT[self._treated] = self._synthetic(self._X_t, self._X_c, self._Y_t, self._Y_c)
			ITT[self._controls] = -self._synthetic(self._X_c, self._X_t, self._Y_c, self._Y_t)

		return Results(ITT[self._treated].mean(), ITT.mean(), ITT[self._controls].mean())


	def lasso_syn(self):

		ITT = np.zeros(self._N_t)
		clf = linear_model.Lasso(alpha=0.00001)

		for i in xrange(self._N_t):
			non_zero = clf.fit(self._X_c.T, self._X_t[i, ]).coef_ > 0  # L1 variable selection
			X_c = self._X_c.T[:, non_zero].T
			X_m1 = np.row_stack((X_c.T, np.ones(X_c.shape[0])))  # add row of 1's
			x1 = np.append(self._X_t[i], 1)  # append 1 to restrict weights to sum to 1
			w = np.linalg.lstsq(X_m1, x1)[0]
			ITT[i] = self._Y_t[i] - np.dot(w, self._Y_c[non_zero])

		return ITT.mean()


	def _norm(self, dX, W):

		"""
		Calculates vector of norms given weighting matrix W.

		Arguments
		---------
			dX: array-like
				Matrix of covariate differences.
			W: array-like
				Weighting matrix to be used in norm calcuation. Acceptable
				values are None	(inverse variance, default), string 'maha'
				for Mahalanobis	metric, or any arbitrary k-by-k matrix.

		Returns
		-------
			Vector of distance measures.
		"""

		if W is None:
			return (dX**2 / self._Xvar).sum(axis=1)
		else:
			return (dX.dot(W)*dX).sum(axis=1)


	def _msmallest_with_ties(self, x, m):

		"""
		Finds indices of the m smallest entries in an array. Ties are
		included, so the number of indices can be greater than m. Algorithm
		is of order O(n).

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

<<<<<<< HEAD
		par_indx = np.argpartition(x, m-1)  # partition around mth order stat
		m_indx = list(par_indx[:m])

		for i in xrange(m, len(x)):
			if np.isclose(x[par_indx[i]], x[par_indx[m-1]]):
				m_indx.append(par_indx[i])

		return m_indx
=======
		par_indx = np.argpartition(x, m)  # partition around (m+1)th order stat
		
		if x[par_indx[:m]].max() < x[par_indx[m]]:  # mth < (m+1)th order stat
			return list(par_indx[:m])
		elif x[par_indx[m]] < x[par_indx[m+1:]].min():  # (m+1)th < (m+2)th
			return list(par_indx[:m+1])
		else:  # mth = (m+1)th = (m+2)th, so increment and recurse
			return self._msmallest_with_ties(x, m+2)
>>>>>>> 256933354881976


	def _matchmaking(self, X, X_m, W=None, m=1):

		"""
		Performs nearest-neigborhood matching using specified weighting
		matrix in measuring distance. Ties are included, so the number
		of matches for a given unit can be greater than m.

		Arguments
		---------
			X: array-like
				Observations to find matches for.
			X_m: array-like
				Pool of potential matches.
			m: integer
				The number of units to match to a given subject.
			W: array-like
				Weighting matrix to be used in norm calcuation. Acceptable
				values are None	(inverse variance, default), string 'maha'
				for Mahalanobis	metric, or any arbitrary k-by-k matrix.

		Returns
		-------
			List of matched indices.
		"""

		m_indx = []

		for i in xrange(X.shape[0]):
			m_indx.append(self._msmallest_with_ties(self._norm(X_m - X[i], W), m))

		return m_indx


	def _bias(self, m_indx, Y_m, X_m, X):

		"""
		Estimates bias resulting from imperfect matches using least squares.
		When estimating ATT, regression should use control units. When
		estimating ATC, regression should use treated units.

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


	def _match_counting(self, m_indx_t, m_indx_c):

		"""
		Calculates each unit's contribution in being used as a matching
		unit.

		Arguments
		---------
			m_indx_t: list
				List of indices of control units that are matched to each
				treated	unit. 
			m_indx_c:
				List of indices of treated units that are matched to each
				control unit.

		Returns
		-------
			Vector containing each unit's contribution in matching.
		"""

		count = np.zeros(self._N)
		for i in xrange(self._N_c):
			M = len(m_indx_c[i])
			for j in xrange(M):
				count[self._treated[m_indx_c[i][j]]] += 1./M
		for i in xrange(self._N_t):
			M = len(m_indx_t[i])
			for j in xrange(M):
				count[self._controls[m_indx_t[i][j]]] += 1./M

		return count


	def _conditional_var(self, W, m):

		"""
		Computes unit-level conditional variances. Estimation is done by
		matching treated units with treated units, control units with control
		units, and then calculating sample variances among the matches.

		Arguments
		---------
			W: array-like
				Weighting matrix to be used in norm calcuation. Acceptable
				values are None	(inverse variance), string 'maha' for
				Mahalanobis metric, or any arbitrary k-by-k matrix.
			m: integer
				The number of units to match to a given subject.

		Returns
		-------
			Vector of conditional variances.
		"""

		# m+1 since we include the unit itself in the matching pool as well
		m_indx_t = self._matchmaking(self._X_t, self._X_t, W, m+1)
		m_indx_c = self._matchmaking(self._X_c, self._X_c, W, m+1)

		cond_var = np.empty(self._N)
		for i in xrange(self._N_t):
			cond_var[self._treated[i]] = self._Y_t[m_indx_t[i]].var(ddof=1)
		for i in xrange(self._N_c):
			cond_var[self._controls[i]] = self._Y_c[m_indx_c[i]].var(ddof=1)

		return cond_var


	def matching(self, wmatrix=None, matches=1, correct_bias=False):

		"""
		Estimates average treatment effects using matching with replacement.

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
				Weighting matrix to be used in norm calcuation. Acceptable
				values are None	(inverse variance, default), string 'maha'
				for Mahalanobis	metric, or any arbitrary k-by-k matrix.
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
			self._Xvar = self._X.var(0)  # store vector of covariate variances
		elif wmatrix == 'maha':
			wmatrix = np.linalg.inv(np.cov(self._X, rowvar=False))

		m_indx_t = self._matchmaking(self._X_t, self._X_c, wmatrix, matches)
		m_indx_c = self._matchmaking(self._X_c, self._X_t, wmatrix, matches)

		ITT = np.empty(self._N)
		for i in xrange(self._N_t):
			ITT[self._treated[i]] = self._Y_t[i] - self._Y_c[m_indx_t[i]].mean()
		for i in xrange(self._N_c):
			ITT[self._controls[i]] = self._Y_t[m_indx_c[i]].mean() - self._Y_c[i]

		if correct_bias:
			ITT[self._treated] -= self._bias(m_indx_t, self._Y_c, self._X_c, self._X_t)
			ITT[self._controls] += self._bias(m_indx_c, self._Y_t, self._X_t, self._X_c)

		cond_var = self._conditional_var(wmatrix, matches)
		match_counts = self._match_counting(m_indx_t, m_indx_c)

		var_ATE = ((1+match_counts)**2 * cond_var).sum() / self._N**2
		var_ATT = ((self._D - (1-self._D)*match_counts)**2 * cond_var).sum() / self._N_t**2
		var_ATC = ((self._D*match_counts - (1-self._D))**2 * cond_var).sum() / self._N_c**2

		return Results(ITT[self._treated].mean(), ITT.mean(), ITT[self._controls].mean(),
		               var_ATE, var_ATT, var_ATC)


	def matching_without_replacement(self, wmatrix=None, order_by_pscore=False, correct_bias=False):

		"""
		Estimates average treatment effects using matching without replacement.

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
				Weighting matrix to be used in norm calcuation. Acceptable
				values are None	(inverse variance, default), string 'maha'
				for Mahalanobis	metric, or any arbitrary k-by-k matrix.
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

		if self._N_t > self._N_c:
			raise IndexError('Not enough control units.')

		m_indx = np.zeros((self._N_t, 1), dtype=np.int)

		if wmatrix is None:
			self._Xvar = self._X.var(0)
		elif wmatrix == 'maha':
			wmatrix = np.linalg.inv(np.cov(self._X, rowvar=False))

		if order_by_pscore:
			pscore = sm.Logit(self._D, self._X).fit(disp=False).predict()  # estimate by logit
			order = np.argsort(pscore[self._D==1])[::-1]  # descending pscore order
		else:
			order = xrange(self._N_t)
			
		unmatched = range(self._N_c)
		for i in order:
			dX = self._X_c[unmatched] - self._X_t[i]
			m_indx[i] = unmatched.pop(np.argmin(self._norm(dX, wmatrix)))

		ITT = self._Y_t - self._Y_c[m_indx]

		if correct_bias:
			ITT -= self._bias(list(m_indx), self._Y_c, self._X_c, self._X_t)

		return Results(ITT.mean())


	def ols(self):

		"""
		Estimates average treatment effects using least squares.

		Returns
		-------
			A Results class instance.
		"""

		D = self._D.reshape((self._N, 1))
		dX = self._X - self._X.mean(0)
		DdX = D * dX
		Z = np.column_stack((D, dX, DdX))

		ITT = np.empty(self._N)
		reg = sm.OLS(self._Y, sm.add_constant(Z)).fit()
		ITT[self._treated] = reg.params[1] + np.dot(dX[self._treated], reg.params[-self._k:])
		ITT[self._controls] = reg.params[1] + np.dot(dX[self._controls], reg.params[-self._k:])

		return Results(ITT[self._treated].mean(), ITT.mean(), ITT[self._controls].mean())


class Results(object):


	def __init__(self, ATT, ATE=None, ATC=None, *variances):

		self.ATT, self.ATE, self.ATC = ATT, ATE, ATC
		self.calculated_var = False
		if variances:
			self.calculated_var = True
			self.var_ATT, self.var_ATE, self.var_ATC = variances


	def summary(self):

		if self.ATE:
			print 'Estimated Average Treatment Effect:', self.ATE, sqrt(self.var_ATE) if self.calculated_var else ''
		print 'Estimated Average Treatment Effect on the Treated:', self.ATT, sqrt(self.var_ATT) if self.calculated_var else ''
		if self.ATC:
			print 'Estimated Average Treatment Effect on the Untreated:', self.ATC, sqrt(self.var_ATC) if self.calculated_var else ''









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


def UseLalonde():

	import pandas as pd

	lalonde = pd.read_csv('ldw_exper.csv')

	covariate_list = ['age', 'educ', 'black', 'hisp', 'married',
	                  're74', 're75', 'u74', 'u75']

	# don't know how to not convert to array first
	Y = np.array(lalonde['re78'])
	D = np.array(lalonde['t'])
	X = np.array(lalonde[covariate_list])

	causal = CausalModel(Y, D, X)
	print causal.normalized_diff()
	causal.matching(matches=4).summary()
	causal.matching(matches=1).summary()
	causal.matching(matches=4, correct_bias=True).summary()

