import numpy as np

from covariates import Covariates
from ..utils.tools import cache_readonly
from ..utils.tools import remove


class Basic(object):

	"""
	Basic data setup class. Defines basic variables like sample sizes,
	and passes X matrix to Covariates class. To be inherited by Stratum
	and Causal classes.
	"""

	def __init__(self, Y, D, X):

		self.Y, self.D, self.X = Y, D, X
		self.N, self.K = self.X.shape
		self.controls, self.treated = (self.D==0), (self.D==1)
		self.Y_c, self.Y_t = self.Y[self.controls], self.Y[self.treated]
		self.X_c, self.X_t = self.X[self.controls], self.X[self.treated]
		self.N_t = self.D.sum()
		self.N_c = self.N - self.N_t
		remove(self, '_covariates')


	@cache_readonly
	def covariates(self):

		return Covariates(self)

