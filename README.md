PyNest
======

Python implementation of nested sampling algorithm for Bayesian evidence computation.

This repository is a port of the MATLAB-based matlabmultinest code from Matthew Pitkin and Joe Romano,
based in part on the "mininest.py" code from Issac Trotts, available on John Skilling's website. The
purpose of this code is to reproduce the Bayesian evidence calculation of the frequency model given
as an example in the matlabmultinest code.

The routines here, written in Python, should accept both the prior density and likelihood functions
as arguments, and requires that a function that produces samples from the prior density is also
passed as an argument. The PyNest.nested_sampler function should perform nested sampling with a 
tolerance calculation by using differential evolution (originally by Storn and Price) to generate 
proposals, which are then accepted or rejected using a Metropolis-Hastings rule. This type of algorithm
generates a reversible Markov Chain in order to generate somewhat independent samples from the prior,
depending on how many steps are taken to allow the chain to forget its starting point. Finally, the
sample is accepted or rejected based on a hard cutoff set by the lowest likelihood sample in the 
existing population of samples. 

I have made some modifications to the matlabmultinest code, implementing a more standard 
implementation of differential evolution, which includes dithering to add an extra element of
randomness to the proposal process.
