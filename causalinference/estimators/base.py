from ..utils.tools import Printer


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

		if self._dict['ate_se'] is None:
			ate_se, att_se, atc_se = self._compute_se()
			self._store_se(ate_se, att_se, atc_se)

		p = Printer()
		est = ['ATE', 'ATT', 'ATC']
		se = [self._dict['ate_se'], self._dict['att_se'],
		      self._dict['atc_se']]

		entries = ('', 'Est.', 'S.e.', 'z', 'P>|z|',
		           '[95% Conf. int.]')
		span = [1]*5 + [2]
		etype = ['string']*6
		output = '\n'
		output += p.write_row(entries, span, etype)
		output += p.write_row('-'*p.table_width, [1], ['string'])

		span = [1]*7
		etype = ['string'] + ['float']*6
		for i in xrange(len(est)):
			coef = self._dict[est[i].lower()]
			entries = p._reg_entries(est[i], coef, se[i])
			output += p.write_row(entries, span, etype)

		return output


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

		output = '\n'
		output += 'Treatment Effect Estimates\n\n'

		for method in self._dict.keys():
			if method == 'ols':
				output += method.upper()
			else:
				output += method.title()
			output += self._dict[method].__str__()
			output += '\n'
			
		return output


	def keys(self):

		return self._dict.keys()

