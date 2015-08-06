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

