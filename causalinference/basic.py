import numpy as np


class Basic(object):


	def __init__(self, Y, D, X):

		self.Y, self.D, self.X = Y, D, X
		self.N, self.K = self.X.shape
		self.Y_t, self.Y_c = self.Y[self.D==1], self.Y[self.D==0]
		self.X_t, self.X_c = self.X[self.D==1], self.X[self.D==0]
		self.N_t = self.D.sum()
		self.N_c = self.N - self.N_t


	@property
	def ndiff(self):

		try:
			return self._ndiff
		except AttributeError:
			self._ndiff = self._compute_ndiff()
			return self._ndiff


	def _compute_ndiff(self):

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

		return (self.X_t.mean(0) - self.X_c.mean(0)) / \
		       np.sqrt((self.X_t.var(0) + self.X_c.var(0))/2)


	def _try_del(self, attrstr):
	
		try:
			delattr(self, attrstr)
		except:
			pass

