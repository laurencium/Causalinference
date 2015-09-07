CausalInference
===============

CausalInference is a Python implementation of statistical and econometric methods in the field variously known as Causal Inference, Program Evaluation, and Treatment Effect Analysis.

Work on CausalInference started in 2014 by Laurence Wong as a personal side project. It is distributed under the 3-Clause BSD license.

The most current development version is hosted on GitHub at:
https://github.com/laurencium/causalinference

Package source and binary distribution files are available from PyPi:
https://pypi.python.org/pypi/CausalInference

For an overview of the main features and uses of CausalInference, please refer to:
https://github.com/laurencium/CausalInference/blob/master/docs/tex/vignette.pdf

Main Features
=============

* Assessment of overlap in covariate distributions
* Estimation of propensity score
* Improvement of covariate balance through trimming
* Subclassification on propensity score
* Estimation of treatment effects via matching, blocking, weighting, and least squares

Dependencies
============

* NumPy: 1.8.2 or higher
* SciPy: 0.13.3 or higher

Installation
============

CausalInference can be installed using ``pip``: ::

  $ pip install causalinference

and will run provided the necessary dependencies are in place.

Minimal Example
===============

The following illustrates how to create an instance of CausalModel: ::

  >>> from causalinference import CausalModel
  >>> from causalinference.utils import random_data
  >>> Y, D, X = random_data()
  >>> causal = CausalModel(Y, D, X)

Invoking ``help`` on ``causal`` at this point should return a comprehensive listing of all the causal analysis tools available in CausalInference.

