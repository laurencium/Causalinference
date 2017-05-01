try:
	from setuptools import setup
except ImportError:
	from distutils.core import setup

config = {
	'name': 'CausalInference',
	'version': '0.1.2',
	'url': 'https://github.com/laurencium/causalinference',
	'author': 'Laurence Wong',
	'author_email': 'laurencium@gmail.com',
	'packages': ['causalinference', 'causalinference.core',
	             'causalinference.estimators', 'causalinference.utils'],
	'include_package_data': True,
	'package_data': {'causalinference': ['utils/*.txt']},
	'license': 'LICENSE.txt',
	'description': 'Causal Inference in Python',
	'long_description': open('README.rst').read()
}

setup(**config)

