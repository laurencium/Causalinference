
import numpy as np
import random
import cvxpy as cvx
import pandas as pd
from scipy.stats import norm
from sklearn import linear_model


def EstimateWeights(X_c, X_t):

	"""
	Function that estimates synthetic control weights for each treated unit.

	For each treated unit, we find w that minimizes:
		|| w' X_c - X_t ||_2,
	subject to the weights being between 0 and 1, and that they sum to 1.

	Convex optimization problem solved by calling CVX.

	Args:
		X_c = N_c-by-k matrix of control units
		X_t = N_t-by-k matrix of treated units

	Returns:
		W_hat = N_t-by-N_c matrix of estimated weights
	"""

	N_c = len(X_c)
	N_t = len(X_t)

	W_hat = np.zeros(shape=(N_t, N_c))  # matrix of weight estimates

	for i in xrange(N_t):

		# for each treated unit, call CVX to solve optimization problem
		w = cvx.Variable(N_c)
		objective = cvx.Minimize(cvx.sum_squares(X_c.T * w - X_t[i, ]))
		constraints = [cvx.sum_entries(w) == 1]
		prob = cvx.Problem(objective, constraints)
		min_value = prob.solve()

		# constraints = [0 <= w, w <= 1, cvx.sum_entries(w) == 1]
		# note: with Lalonde data, restricting 0 <= w <= 1 yields no
		#       solutions for some treated units

		W_hat[i, ] = w.value.getA1()  # convert to flattened array and store

	return W_hat


def TreatmentEffects(Y_c, Y_t, W, return_individual=False):

	"""
	Function that estimates individual and average treatment effects
	via synthetic control method, using provided synthetic control weights.

	Args:
		Y_c = N_c-dimensional array of control unit outcomes
		Y_t = N_t-dimensional array of treated unit outcomes
		W = N_t-by-N_c matrix of synthetic control weights

	Returns:
		ITT = N_t-dimensional array of estimated individual treatment effects
		      for the treated sample
		ATT = Scalar value of estaimted average treatment effect on the treated
	"""

	ITT = Y_t - np.dot(W, Y_c)
	ATT = ITT.mean()

	if return_individual:
		return ATT, ITT
	else:
		return ATT


def SyntheticEstimates(Y, D, X, higher_moments=False):

	"""
	Function that computes ATT estimates from observerd data using
	synthetic control method.
	"""

	control = (D == 0)  # Boolean index of control units
	treated = (D == 1)  # Boolean index of treated units
	if higher_moments:
		X_control = np.hstack((X[control], abs(X[control]), X[control]**2))
		X_treated = np.hstack((X[treated], abs(X[treated]), X[treated]**2))
		W_hat = EstimateWeights(X_control, X_treated)
	else:
		W_hat = EstimateWeights(X[control], X[treated])  # synthetic control weights

	return TreatmentEffects(Y[control], Y[treated], W_hat)


def MatchWithReplacement(Y_c, Y_t, X_c, X_t):

	"""
	Function that estimates the average treatment effect for the treated (ATT)
	using matching done with replacement.

	Args:
		Y_c = N_c-dimensional array of control unit outcomes
		Y_t = N_t-dimensional array of treated unit outcomes
		X_c = N_c-by-k matrix of control units
		X_t = N_t-by-k matrix of treated units

	Returns:
		ATT estimates; matching done with replacement
	"""

	N_c = len(X_c)
	N_t = len(X_t)

	ITT = np.zeros(N_c)

	for i in xrange(N_c):
		# find control unit that minimizes L_2-norm between X_c and X_t
		# min search should be at most linear time
		match_index = ((X_t - X_c[i])**2).sum(1).argmin()
		ITT[i] = Y_t[match_index] - Y_c[i]  # estimated individual treatment effect

	return ITT.mean()


def MatchingEstimates(Y, D, X, with_replacement=True):

	"""
	Function that estimates the average treatment effect for the treated (ATT)
	using matching.

	Can specify whether matching is done with or without replacement (not yet).

	Args:
		Y = N-dimensional array of observed outcomes
		D = N-dimensional array of treatment indicator; 1=treated, 0=control
		X = N-by-k matrix of covariates

	Returns:
		ATT estimates; matching done with or without replacement as specified
	"""

	control = (D == 0)  # Boolean index of control units
	treated = (D == 1)  # Boolean index of treated units

	# put replacement conditional here later
	return MatchWithReplacement(Y[control], Y[treated], X[control], X[treated])


def OLSEstimates(Y, D, X):

	"""
	Function that estimates average treatment effect for the treated (ATT)
	using OLS, which is consistent when the true model is linear.

	Args:
		Y = N-dimensional array of observed outcomes
		D = N-dimensional array of treatment indicator; 1=treated, 0=control
		X = N-by-k matrix of covariates
	"""

	k = X.shape[1]
	D = D.reshape((D.size, 1))  # convert D into N-by-1 vector
	dX = X - X.mean(0)  # demean covariates
	DdX = D * dX
	# construct design matrix; no constant term as it will be added by sklearn
	W = np.column_stack((D, dX, DdX))

	# use sklearn to run regression
	reg = linear_model.LinearRegression()
	reg.fit(W, Y)

	# for derivation of this estimator, see my notes on Treatment Effects
	return reg.coef_[0] + np.dot((DdX.sum(0) / D.sum()), reg.coef_[-k:])


class parameters:

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


def MonteCarlo(B=500, para=parameters(), nonlinear=False, print_progress=True):

	"""
	Function that returns ATT estimates using synthetic control, matching,
	and OLS computed over B repetitions on simulated data.

	See SimulateData function for data generating process.

	Args:
		B = Number of Monte Carlo simulations to perform
		para = Model parameter values supplied by parameter class object
		nonlinear = Boolean indicating whether the data generating model should
		            be linear or not
		print_progress = Boolean indicating whether to dispaly current number
		                 of completed repetitions (every 10 repetitions)

	Returns:
		ATT_true = Actual average treatment effect for the treated
		ATT_syn = Estimated ATT using synthetic controls
		ATT_match = Estimated ATT using matching
		ATT_ols = Estimated ATT using OLS
	"""

	ATT_true = np.zeros(B)
	ATT_syn = np.zeros(B)
	ATT_match = np.zeros(B)
	ATT_ols = np.zeros(B)

	for i in xrange(B):

		Y, D, X, Y_0, Y_1 = SimulateData(para, nonlinear, True)

		ATT_true[i] = (Y_1[D==1]-Y_0[D==1]).mean()
		ATT_syn[i] = SyntheticEstimates(Y, D, X, higher_moments=nonlinear)
		ATT_match[i] = MatchingEstimates(Y, D, X)
		ATT_ols[i] = OLSEstimates(Y, D, X)

		if print_progress and (i+1) % 10 == 0:
			print i+1, 'out of', B, 'simulations completed.'

	return ATT_true, ATT_syn, ATT_match, ATT_ols


def CalculateMSE(B=500, para=parameters(), nonlinear=False):

	"""
	Function that calcuates MSE after performing Monte Carlo simulations.

	Args:
		B = Number of Monte Carlo simulations to perform
		para = Model parameter values supplied by parameter class object
		nonlinear = Boolean indicating whether the data generating model should
		            be linear or not

	Returns:
		MSE_syn = Estimated MSE using synthetic control estimates
		MSE_match = Estimated MSE using matching estimates
		MSE_ols = Estimated MSE using OLS estimates
	"""

	ATT_true, ATT_syn, ATT_match, ATT_ols = MonteCarlo(B, nonlinear=nonlinear)

	MSE_syn = ((ATT_syn - ATT_true)**2).mean()
	MSE_match = ((ATT_match - ATT_true)**2).mean()
	MSE_ols = ((ATT_ols - ATT_true)**2).mean()

	return MSE_syn, MSE_match, MSE_ols


def UseLalondeData():

	"""
	Utility function that reads and applies synthetic control
	estimator on Lalonde's NSW experimental data.

	Note that because of complete randomization, the mean
	difference identifies both the ATE and ATT.
	"""

	url = 'http://www.stanford.edu/~lwong1/data.csv'
	lalonde = pd.read_csv(url)  # read CSV data from url

	covariate_list = ['age', 'education', 'black', 'hispanic', 'married',
	                  'nodegree', 're74', 're75', 'u74', 'u75']
	treated = (lalonde['treat'] == 1)  # index of treated units
	control = (lalonde['treat'] == 0)  # index of control units

	# don't know how to not convert to array first
	X_t = np.array(lalonde[treated][covariate_list])
	X_c = np.array(lalonde[control][covariate_list])
	W_hat = EstimateWeights(X_c, X_t)

	Y_t = np.array(lalonde[treated]['re78'])
	Y_c = np.array(lalonde[control]['re78'])
	ATT_hat = TreatmentEffects(Y_c, Y_t, W_hat)

	print "Using Lalonde's National Supported Work (NSW) experimental data..."
	print 'Mean difference:', Y_t.mean() - Y_c.mean()
	print 'Estimated average treatment effect on the treated:', ATT_hat


def main():

	UseLalondeData()

	B = 50
	print '\n' + 'Performing Monte Carlo simulations with B =', B
	MSE_syn, MSE_match, MSE_ols = CalculateMSE(B, nonlinear=True)
	print 'Estimated Mean Squared Error for synthetic control estimator:', MSE_syn
	print 'Estimated Mean Squared Error for matching estimator:', MSE_match
	print 'Estimated Mean Squared Error for OLS estimator:', MSE_ols


if __name__ == '__main__':
	main()
