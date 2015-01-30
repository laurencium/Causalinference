import numpy as np

from .covariates import Covariates
from utils.tools import cache_readonly
from utils.tools import _try_del


class Basic(object):


	def __init__(self, Y, D, X):

		self.Y, self.D, self.X = Y, D, X
		self.N, self.K = self.X.shape
		self.Y_t, self.Y_c = self.Y[self.D==1], self.Y[self.D==0]
		self.X_t, self.X_c = self.X[self.D==1], self.X[self.D==0]
		self.N_t = self.D.sum()
		self.N_c = self.N - self.N_t
		_try_del(self, '_covariates')


	@cache_readonly
	def covariates(self):

		return Covariates(self.X_c, self.X_t)



