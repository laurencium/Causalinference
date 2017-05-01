Causal Inference in Python
==========================

*Causal Inference in Python*, or *Causalinference* in short, is a software package that implements various statistical and econometric methods used in the field variously known as Causal Inference, Program Evaluation, or Treatment Effect Analysis.

Work on *Causalinference* started in 2014 by Laurence Wong as a personal side project. It is distributed under the 3-Clause BSD license.

Important Links
===============

The official website for *Causalinference* is

  http://causalinferenceinpython.org

The most current development version is hosted on GitHub at

  https://github.com/laurencium/causalinference

Package source and binary distribution files are available from PyPi at

  https://pypi.python.org/pypi/causalinference

For an overview of the main features and uses of *Causalinference*, please refer to

  https://github.com/laurencium/causalinference/blob/master/docs/tex/vignette.pdf

A blog dedicated to providing a more detailed walkthrough of *Causalinference* and the econometric theory behind it can be found at

  http://laurence-wong.com/software/

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

*Causalinference* can be installed using ``pip``: ::

  $ pip install causalinference

For help on setting up Pip, NumPy, and SciPy on Macs, check out this excellent `guide <http://www.sourabhbajaj.com/mac-setup>`_.

Minimal Example
===============

The following illustrates how to create an instance of CausalModel: ::

  >>> from causalinference import CausalModel
  >>> from causalinference.utils import random_data
  >>> Y, D, X = random_data()
  >>> causal = CausalModel(Y, D, X)

Invoking ``help`` on ``causal`` at this point should return a comprehensive listing of all the causal analysis tools available in *Causalinference*.

