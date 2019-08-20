# swpr
Variational inference for Wishart and inverse Wishart processes

Implementation of the models described in "Large-scale Bayesian dynamic covariance modeling with variational Wishart and inverse Wishart processes" (Heaukulani & van der Wilk, 2019). The paper can be found at:

https://arxiv.org/abs/1906.09360

The implementation relies on [GPflow](https://www.gpflow.org), and the model is actually just a thin wrapper around the SVGP model class in GPflow. The 'models.py' submodule implements the wrappers around SVGP, and the 'likelihoods.py' submodule computes the (Monte Carlo estimate of) the likehood function, which is the only line that really needs to be changed in this black-box implementation (see the paper).

Investigate the file 'demo.py', which you can run as a script to see usage. This demo makes use of [GPflow Monitor](https://gpflow.readthedocs.io/en/develop/notebooks/advanced/monitoring.html?highlight=monitor)'s functionality, though it should be clear how to manage tracking and saving/loading via classic Tensorflow.

This code was built on:
* tensorflow==1.12.0
* gpflow==1.3.0
