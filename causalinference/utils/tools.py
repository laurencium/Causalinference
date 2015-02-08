import numpy as np
from scipy.stats import norm


def cache_readonly(func):

	def try_cache(*args):

		try:
			return getattr(args[0], '_'+func.__name__)
		except AttributeError:
			setattr(args[0], '_'+func.__name__, func(*args))
			return getattr(args[0], '_'+func.__name__)

	return property(try_cache)


def remove(obj, attrstr):

	if hasattr(obj, attrstr):
		delattr(obj, attrstr)


class Printer(object):

	def __init__(self):

		self.table_width = 80


	def _reg_entries(self, name, coef, se):

		"""
			Constructs a tuple of derived entries given regression
			coefficient estimates.

		Expected args
		-------------
			name: string
				Variable name associated with the estimated
				coefficient.
			coef: float
				Estimated coefficient.
			se: float
				Standard error associated with the estimated
				coefficient.

		Returns
		-------
			Tuple containing name, coefficient estimate, standard
			error, z-statistic, p-value, and lower and upper limit
			of the corresponding 95% confidence interval.
		"""

		z = coef / se
		p = 1 - norm.cdf(np.abs(z))
		lw = coef - 1.96*se
		up = coef + 1.96*se

		return (name, coef, se, z, p, lw, up)


	def write_row(self, entries, span, etype):

		"""
		Constructs string for a row in a table given desired entries
		in the row, and formatting options as specified by the other
		arguments.

		Expected args
		-------------
			entries: tuple
				Entries in the row to print.
			span: list
				Corresponding to each entry, the number of
				columns the entry should span.
			etype: list
				Corresponding to each entry, the type of
				entry (e.g., 'string', 'float', or 'integer').

		Returns
		-------
			String containing the entries with the right format
			as specified by the other arguments.
		"""

		k = len(span)
		cols = sum(span)
		span = [(self.table_width//cols) * span[i] for i in xrange(k)]
		span[0] += self.table_width % cols
		for i in xrange(k):
			if etype[i] == 'string':
				etype[i] = 's'
			elif etype[i] == 'float':
				etype[i] = '.3f'
			elif etype[i] == 'integer':
				etype[i] = '.0f'

		line = ['%'+str(span[i])+etype[i] for i in xrange(k)]

		return (''.join(line) % entries) + '\n'

