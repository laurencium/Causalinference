import causalinference.utils.tools as tools
from ..core import Dict


class Estimator(Dict):

	"""
	Dictionary-like class containing treatment effect estimates.
	"""

	def __str__(self):

		table_width = 80

		names = ['ate', 'atc', 'att']
		coefs = [self[name] for name in names if name in self.keys()]
		ses = [self[name+'_se'] for name in names if name+'_se' in self.keys()]

		output = '\n'
		output += 'Treatment Effect Estimates: ' + self._method + '\n\n'

		entries1 = ['', 'Est.', 'S.e.', 'z', 'P>|z|',
		           '[95% Conf. int.]']
		entry_types1 = ['string']*6
		col_spans1 = [1]*5 + [2]
		output += tools.add_row(entries1, entry_types1,
		                        col_spans1, table_width)
		output += tools.add_line(table_width)

		entry_types2 = ['string'] + ['float']*6
		col_spans2 = [1]*7
		for (name, coef, se) in zip(names, coefs, ses):
			entries2 = tools.gen_reg_entries(name.upper(), coef, se)
			output += tools.add_row(entries2, entry_types2,
			                        col_spans2, table_width)

		return output


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


	def __iter__(self):

		return iter(self._dict)


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

