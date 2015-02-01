from utils.tools import cache_readonly


class Estimates(object):

	def __init__(self, ate, att, atc, method, obj):

		self.ate, self.att, self.atc = ate, att, atc
		self._method = method
		self.obj = obj


	def __dir__(self):

		return ['ate', 'att', 'atc', 'ate_se', 'att_se', 'atc_se']


	@cache_readonly
	def _se(self):

		return self.obj._compute_se(self._method) 


	@property
	def ate_se(self):

		return self._se[0]


	@property
	def att_se(self):

		return self._se[1]


	@property
	def atc_se(self):

		return self._se[2]
