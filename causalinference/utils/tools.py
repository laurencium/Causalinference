import numpy as np
from scipy.stats import norm


def convert_to_formatting(entry_types):

	for entry_type in entry_types:
		if entry_type == 'string':
			yield 's'
		elif entry_type == 'float':
			yield '.3f'
		elif entry_type == 'integer':
			yield '.0f'


def add_row(entries, entry_types, col_spans, width):

	"""
	Convert an array of string or numeric entries into a string with
	even formatting and spacing.
	"""

	vis_cols = len(col_spans)
	invis_cols = sum(col_spans)

	char_per_col = width // invis_cols
	first_col_padding = width % invis_cols

	char_spans = [char_per_col * col_span for col_span in col_spans]
	char_spans[0] += first_col_padding
	formatting = convert_to_formatting(entry_types)
	line = ['%'+str(s)+f for (s,f) in zip(char_spans,formatting)]

	return (''.join(line) % tuple(entries)) + '\n'


def add_line(width):

	return '-'*width + '\n'


def gen_reg_entries(varname, coef, se):

	z = coef / se
	p = 2*(1 - norm.cdf(np.abs(z)))
	lw = coef - 1.96*se
	up = coef + 1.96*se

	return (varname, coef, se, z, p, lw, up)


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

