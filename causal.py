import numpy as np
from scipy.optimize import fmin_bfgs


class CausalModel(object):


	def __init__(self, Y, D, X):

		self._Y, self._D, self._X = Y, D, X
		self._N, self._K = self._X.shape
		self._treated, self._controls = np.nonzero(self._D)[0], np.nonzero(self._D==0)[0]
		self._Y_t, self._Y_c = self._Y[self._treated], self._Y[self._controls]
		self._X_t, self._X_c = self._X[self._treated], self._X[self._controls]
		self._N_t = np.count_nonzero(self._D)
		self._N_c = self._N - self._N_t
		self._normalized_diff = None
		self._pscore = {}
		self.restore()


	def restore(self):

		self.Y, self.D, self.X = self._Y, self._D, self._X
		self.N, self.K = self._N, self._K
		self.treated, self.controls = self._treated, self._controls
		self.Y_t, self.Y_c = self._Y_t, self._Y_c
		self.X_t, self.X_c = self._X_t, self._X_c
		self.N_t, self.N_c = self._N_t, self._N_c
		self.normalized_diff = self._normalized_diff
		self.pscore = self._pscore


	def compute_normalized_diff(self):

		"""
		Computes normalized difference in covariates for assessing balance.

		Normalized difference is the difference in group means, scaled by the
		square root of the average of the two within-group variances. Large
		values indicate that simple linear adjustment methods may not be adequate
		for removing biases that are associated with differences in covariates.

		Unlike t-statistic, normalized differences do not, in expectation,
		increase with sample size, and thus is more appropriate for assessing
		balance.

		Returns
		-------
			Vector of normalized differences.
		"""

		self.normalized_diff = (self.X_t.mean(0) - self.X_c.mean(0)) / \
		                       np.sqrt((self.X_t.var(0) + self.X_c.var(0))/2)


	def _sigmoid(self, x):
	
		"""
		Computes 1/(1+exp(-x)) for input x, to be used in maximum likelihood
		estimation of propensity score.

		Arguments
		---------
			x: array-like

		Returns
		-------
			Vector or scalar 1/(1+exp(-x)), depending on input x.
		"""

		return 1/(1+np.exp(-x))


	def _log1exp(self, x):

		"""
		Computes log(1+exp(-x)) for input x, to be used in maximum likelihood
		estimation of propensity score.

		Arguments
		---------
			x: array-like

		Returns
		-------
			Vector or scalar log(1+exp(-x)), depending on input x.
		"""

		return np.log(1 + np.exp(-x))


	def _neg_loglike(self, beta, X_t, X_c):

		"""
		Computes the negative of the log likelihood function for logit, to be used
		in maximum likelihood estimation of propensity score. Negative because SciPy
		optimizier does minimization only.

		Arguments
		---------
			beta: array-like
				Logisitic regression parameters to maximize over.
			X_t: array-like
				Covariate matrix of the treated units.
			X_c: array-like
				Covariate matrix of the control units.

		Returns
		-------
			Negative log likelihood evaluated at input values.
		"""

		return self._log1exp(X_t.dot(beta)).sum() + \
		       self._log1exp(-X_c.dot(beta)).sum()


	def _neg_gradient(self, beta, X_t, X_c):

		"""
		Computes the negative of the gradient of the log likelihood function for
		logit, to be used in maximum likelihood estimation of propensity score.
		Negative because SciPy optimizier does minimization only.

		Arguments
		---------
			beta: array-like
				Logisitic regression parameters to maximize over.
			X_t: array-like
				Covariate matrix of the treated units.
			X_c: array-like
				Covariate matrix of the control units.

		Returns
		-------
			Negative gradient of log likelihood function evaluated at input values.
		"""

		return (self._sigmoid(X_c.dot(beta))*X_c.T).sum(1) - \
		       (self._sigmoid(-X_t.dot(beta))*X_t.T).sum(1)


	def _compute_propensity_score(self, X):

		"""
		Estimates via logit the propensity score based on input covariate matrix X.

		Arguments
		---------
			X: array-like
				Covariate matrix to estimate propensity score on.

		Returns
		-------
			beta: array-like
				Estimated logistic regression coefficients.
			loglike: scalar
				Maximized log-likelihood value.
			fitted: array-like
				Estimated propensity scores for each unit.
		"""

		X_t = X[self.treated]
		X_c = X[self.controls]
		K = X.shape[1]

		neg_loglike = lambda x: self._neg_loglike(x, X_t, X_c)
		neg_gradient = lambda x: self._neg_gradient(x, X_t, X_c)

		logit = fmin_bfgs(neg_loglike, np.zeros(K), neg_gradient, full_output=True, disp=False)

		pscore = {}
		pscore['coeff'] = logit[0]
		pscore['loglike'] = -logit[1]
		pscore['fitted'] = np.empty(self.N)
		pscore['fitted'][self.treated] = self._sigmoid(X_t.dot(pscore['coeff']))
		pscore['fitted'][self.controls] = self._sigmoid(X_c.dot(pscore['coeff']))

		return pscore


	def _form_matrix(self, const, lin, qua):

		"""
		Forms covariate matrix for use in propensity score estimation, based on
		requirements on constant term, linear terms, and quadratic terms.

		Arguments
		---------
			const: Boolean
				Includes a column of one's if True.
			lin: list
				Column numbers of the base self._X covariate matrix
				to include linearly.
			qua: list
				Tuples indicating which columns of the base self._X
				covariate matrix to multiply and include. E.g.,
				[(0,0), (1,2)] indicates squaring the 0th column and
				including the product of the 1st and 2nd columns.

		Returns
		-------
			mat: array-like
				Covariate matrix formed based on requirements on
				constant, linear, and quadratic terms.
		"""

		mat = np.empty((self.N, const+len(lin)+len(qua)))

		current_col = 0
		if const:
			mat[:, current_col] = 1
			current_col += 1
		if lin:
			mat[:, current_col:current_col+len(lin)] = self.X[:, lin]
			current_col += len(lin)
		for term in qua:
			mat[:, current_col] = self.X[:, term[0]] * self.X[:, term[1]]
			current_col += 1

		return mat


	def compute_propensity_score(self, const=True, lin=None, qua=[]):

		"""
		Estimates via logit the propensity score based on requirements on
		constant term, linear terms, and quadratic terms.

		Arguments
		---------
			const: Boolean
				Includes a column of one's if True. Defaults to
				True.
			lin: list
				Column numbers of the base covariate matrix X
				to include linearly. Defaults to using the
				whole covariate matrix.
			qua: list
				Column numbers of the base covariate matrix X
				to include quadratic. E.g., [0,2] will include
				squares of the 0th and 2nd columns, and the product
				of these two columns. Default is not include any
				quadratic terms.

		Returns
		-------
			beta: array-like
				Estimated logistic regression coefficients.
			loglike: scalar
				Maximized log-likelihood value.
			fitted: array-like
				Estimated propensity scores for each unit.
		"""

		if lin is None:
			lin = xrange(self.X.shape[1])
		if qua:
			qua = list(itertools.combinations_with_replacement(qua, 2))

		self.pscore = self._compute_propensity_score(self._form_matrix(const, lin, qua))
		

	def _pscore_select(self, const, X_cur, X_pot, crit, X_lin=[]):
	
		"""
		Estimates via logit the propensity score using Imbens and Rubin's
		covariate selection algorithm.

		Arguments
		---------
			const: Boolean
				Includes a column of one's if True. 
			X_cur: list
				List containing terms that are currently included
				in the logistic regression.
			X_pot: list
				List containing candidate terms to be iterated through.
			crit: scalar
				Critical value used in likelihood ratio test to decide
				whether candidate terms should be included.
			X_lin: list
				List containing linear terms that have been decided on.
				If non-empty, then X_cur and X_pot should be containing
				candidate quadratic terms. If empty, then those two
				matrices should be containing candidate linear terms.

		Returns
		-------
			List containing terms that the algorithm has settled on for inclusion.
		"""

		if not X_pot:
			return X_cur

		if not X_lin:  # X_lin is empty, so linear terms not yet decided
			ll_null = self._pscore(self._form_matrix(const, X_cur, []))[1]
		else:  # X_lin is not empty, so linear terms are already fixed
			ll_null = self._pscore(self._form_matrix(const, X_lin, X_cur))[1]

		lr = np.empty(len(X_pot))
		if not X_lin:
			for i in xrange(len(X_pot)):
				lr[i] = 2*(self._pscore(self._form_matrix(const, X_cur+[X_pot[i]], []))[1] - ll_null)
		else:
			for i in xrange(len(X_pot)):
				lr[i] = 2*(self._pscore(self._form_matrix(const, X_lin, X_cur+[X_pot[i]]))[1] - ll_null)

		argmax = np.argmax(lr)
		if lr[argmax] < crit:
			return X_cur
		else:
			new_term = X_pot.pop(argmax)
			return self._pscore_select(const, X_cur+[new_term], X_pot, crit, X_lin)


	def pscore_select(self, const=True, X_B=[], C_lin=1, C_qua=2.71):

		"""
		Estimates via logit the propensity score using Imbens and Rubin's
		covariate selection algorithm.

		Arguments
		---------
			const: Boolean
				Includes a column of one's if True. Defaults to
				True.
			X_B: list
				Column numbers of the base covariate matrix X
				that should be included as linear terms
				regardless. Defaults to empty list, meaning
				every column of X is subjected to the selection
				algorithm.
			C_lin: scalar
				Critical value used in likelihood ratio test to decide
				whether candidate linear terms should be included.
				Defaults to 1 as in Imbens (2014).
			C_qua: scalar
				Critical value used in likelihood ratio test to decide
				whether candidate quadratic terms should be included.
				Defaults to 2.71 as in Imbens (2014).

		Returns
		-------
			beta: array-like
				Estimated logistic regression coefficients.
			loglike: scalar
				Maximized log-likelihood value.
			fitted: array-like
				Estimated propensity scores for each unit.

		References
		----------
			Imbens, G. & Rubin, D. (2015). Causal Inference in Statistics,
				Social, and Biomedical Sciences: An Introduction.
			Imbens, G. (2014). Matching Methods in Practice: Three Examples.
		"""

		if C_lin == 0:
			X_lin = xrange(self._X.shape[1])
		else:
			X_pot = list(set(xrange(self._X.shape[1])) - set(X_B))
			X_lin = self._pscore_select(const, X_B, X_pot, C_lin)

		if C_qua == np.inf:
			X_qua = []
		else:
			X_pot = list(itertools.combinations_with_replacement(X_lin, 2))
			X_qua = self._pscore_select(const, [], X_pot, C_qua, X_lin)

		return self._pscore(self._form_matrix(const, X_lin, X_qua))


class Results(object):

	def __init__(self, causal):

		self.causal = causal

	def normalized_diff(self):

		if not self.causal.normalized_diff:
			self.causal.compute_normalized_diff()

		print self.causal.normalized_diff


	def propensity_score(self):

		if not self.causal.pscore:
			self.causal.compute_propensity_score()

		print 'Coefficients:', self.causal.pscore['coeff']
		print 'Log-likelihood:', self.causal.pscore['loglike']

from scipy.stats import norm
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


Y, D, X = SimulateData()
causal = CausalModel(Y, D, X)
display = Results(causal)
