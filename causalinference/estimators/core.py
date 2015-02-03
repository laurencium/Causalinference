

class Estimator(object):

	"""
	Dictionary-like class containing treatment effect estimates associated
	with one method (e.g., OLS, matching, etc.). Standard errors for most
	estimators are computed only when needed.
	"""

	def __init__(self):

		ate, att, atc = self._compute_est()
		self._dict = {'ate': ate, 'att': att, 'atc': atc,
		              'ate_se': None, 'att_se': None, 'atc_se': None}

	
	def __repr__(self):

		if self._dict['ate_se'] is None:
			ate_se, att_se, atc_se = self._compute_se()
			self._store_se(ate_se, att_se, atc_se)

		return repr(self._dict)


	def __str__(self):

		return "Print estimates in table here."


	def __getitem__(self, key):

		if 'se' in key and self._dict['ate_se'] is None:
			ate_se, att_se, atc_se = self._compute_se()
			self._store_se(ate_se, att_se, atc_se)

		return self._dict[key]


	def _store_se(self, ate_se, att_se, atc_se):

		self._dict['ate_se'] = ate_se
		self._dict['att_se'] = att_se
		self._dict['atc_se'] = atc_se


	def keys(self):

		return self._dict.keys()


class Estimators(object):

	"""
	Dictionary-like class containing treatment effect estimates for each
	estimator used.
	"""

	def __init__(self):

		self._dict = {}


	def __getitem__(self, key):

		return self._dict[key]


	def __setitem__(self, key, value):

		self._dict[key] = value


	def __str__(self):

		return "Print all estimates in table here."


	def keys(self):

		return self._dict.keys()

