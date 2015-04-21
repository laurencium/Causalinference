try:
	from setuptools import setup
except ImportError:
	from distutils.core import setup

config = {
	'name': 'CausalInference',
	'version': '0.0.4',
	'url': 'https://github.com/laurencium/CausalInference',
	'author': 'Laurence Wong',
	'author_email': 'laurencium@gmail.com',
	'packages': ['causalinference', 'causalinference.core',
	             'causalinference.estimators', 'causalinference.utils'],
	'license': 'LICENSE.txt',
	'description': 'Causal Inference for Python',
	'long_description': open('README.rst').read()
}

setup(**config)

