import numpy as np


class Data(object):

	"""
	Dictionary-like class containing basic data.
	"""

	def __init__(self, Y, D, X):

		self._dict = dict()
		self._dict['Y'] = Y
		self._dict['D'] = D.astype(int)
		self._dict['X'] = X
		self._dict['N'], self._dict['K'] = X.shape
		self._dict['controls'] = (D==0)
		self._dict['treated'] = (D==1)
		self._dict['Y_c'] = Y[self._dict['controls']]
		self._dict['Y_t'] = Y[self._dict['treated']]
		self._dict['X_c'] = X[self._dict['controls']]
		self._dict['X_t'] = X[self._dict['treated']]
		self._dict['N_t'] = D.sum()
		self._dict['N_c'] = self._dict['N'] - self._dict['N_t']


	def __getitem__(self, key):

		return self._dict[key]


	def __setitem__(self, key, value):

		if key == 'pscore':
			self._dict[key] = value
		else:
			raise TypeError("'" + self.__class__.__name__ +
			                "' object does not support item " +
					"assignment")


	def __iter__(self):

		return iter(self._dict)


	def __repr__(self):

		return self._dict.__repr__()


	def keys(self):

		return self._dict.keys()

