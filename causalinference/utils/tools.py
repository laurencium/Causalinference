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


	def print_row(self, entries, span, etype):

		"""
		Expected args
		-------------
			entries: tuple
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
			else:
				etype[i] = '.0f'

		line = ['%'+str(span[i])+etype[i] for i in xrange(k)]
		print ''.join(line) % entries


	def print_table(self, entries, span, etype):

		n = len(entries)
		for i in xrange(n):
			self.print_row(entries[i], span[i], etype[i])

