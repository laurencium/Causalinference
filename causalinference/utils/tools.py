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


	def print_sep(self, symbol):

		print symbol * self.table_width


	def print_row(self, entries, span, string):

		"""
		Expected args
		-------------
			entries: tuple
		"""

		l = len(span)
		cols = sum(span)
		span = [(self.table_width//cols) * span[i] for i in xrange(l)]
		span[0] += self.table_width % cols
		for i in xrange(l):
			if string[i]:
				string[i] = 's'
			else:
				string[i] = '.3f'

		line = ['%'+str(span[i])+string[i] for i in xrange(l)]
		print ''.join(line) % entries

