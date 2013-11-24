# PyNest - a Python implementation of nested sampling, based on matlab multinest

import numpy

def draw_mcmc(livepoints, cholmat, logLmin,
    prior, data, likelihood, model, Nmcmc, parnames, extraparvals):

    mcmcfrac = 0.9
    l2p = 0.5*log(2*numpy.pi) # useful constant

    Nlive = livepoints.shape[0]
    Npars = livepoints.shape[1]

    Ndegs = 2 # student's t distribution number of degrees of freedom

    # initialize counters
    acctot = 0
    Ntimes = 1

    while True:
        acc = 0

        # get random point from live point array
        sampidx = numpy.ceil(numpy.random.rand(1)*Nlive)
        sample = livepoints[sampidx,:]

        # get the sample prior
        # replace this later -- it should compute the prior at the location of
        # "sample"; probably most general to have the user pass a function that
        # we can evalute here to do this.
        p_ub = 10
        p_lb = 0
        currentPrior = -numpy.log(p_ub - p_lb)
        # In fact, let's just write out the code relying on lambda.
        curentPrior = prior(sample)

        for i in range(0,Nmcmc-1):
            if numpy.random.rand(1) < mcmcfrac: # use Student t proposal
                # Draw points from multivariate Gaussian
                gasdevs = numpy.random.randn(Npars)
                sampletmp = (cholmat*gasdevs)

                # calculate chi-square distributed value
                chi = numpy.sum(numpy.power(numpy.random.randn(Ndegs),2))

                # add value onto old sample
                sampletmp = sample + sampletmp*numpy.sqrt(Ndegs/chi)
            else: # use differential evolution
                # first, select a random index
                idx1 = numpy.ceil(numpy.random.rand(1)*Nlive)
                idx2 = idx1.copy()

                # now ensure that the indices are distinct from each other
                # This step is modified from Pitkin; I think all three
                # indices must be distinct
                while idx2 == idx1 or idx1 == sampidx or idx2 == sampidx:
                    idx2 = ceil(numpy.random.rand(1)*Nlive)

                # select the points corresponding to the indices
                A = livepoints[idx1,:]
                B = livepoints[idx2,:]

                # Define differential evolution constants
                F = 1.2 # F = 1.2 stretches the distribution of points on avg
                CR = 0.9 # Crossover probability for each dimension

                sampletmp = sample + (B-A)


    return (sample, logL)

def logplus(logx, logy):

# Copied essentially verbatim from Matlab MultiNest code by Pitkin et al
# logz = logplus(logx, logy)
#
# Given logx and logy, this function returns logz=log(x+y).
# It avoids problems of dynamic range when the
# exponentiated values of x or y are very large or small.


    if numpy.isinf(logx) and numpy.isinf(logy):
        logz = -inf
        return logz


    if logx > logy:
        logz = logx+numpy.log(1.+numpy.exp(logy-logx));
    else:
        logz = logy+numpy.log(1.+numpy.exp(logx-logy));


    return logz
