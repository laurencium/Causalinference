try:
	from setuptools import setup
except ImportError:
	from distutils.core import setup

config = {
	'name': 'Causalinference',
	'version': '0.0.7',
	'url': 'https://github.com/laurencium/causalinference',
	'author': 'Laurence Wong',
	'author_email': 'laurencium@gmail.com',
	'packages': ['causalinference', 'causalinference.core',
	             'causalinference.estimators', 'causalinference.utils'],
	'license': 'LICENSE.txt',
	'description': 'Causal Inference in Python',
	'long_description': open('README.rst').read()
}

setup(**config)

