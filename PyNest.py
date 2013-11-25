# PyNest - a Python implementation of nested sampling, based on matlab multinest

import numpy

def draw_mcmc(livepoints, cholmat, logLmin,
    prior, prior_bounds, data, likelihood, model, Nmcmc, parnames, extraparvals):

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
        # In fact, let's just write out the code relying on lambda. This 'prior'
        # function should actually return the natural logarithm of the prior.
        currentPrior = prior(sample)

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

                # first, select three random indices
                idx1 = numpy.ceil(numpy.random.rand(1)*Nlive)
                idx2 = numpy.ceil(numpy.random.rand(1)*Nlive)
                idx3 = numpy.ceil(numpy.random.rand(1)*Nlive)

                # keep drawing to make sure it's distinct from sampidx
                while idx1 == sampidx:
                    idx1 = numpy.ceil(numpy.random.rand(1)*Nlive)



                # now ensure that the indices are distinct from each other
                # This step is modified from Pitkin; I think all three
                # indices must be distinct. We also use three samples other
                # than the current sample itself, versus two
                while idx2 == idx1 or idx2 == sampidx:
                    idx2 = ceil(numpy.random.rand(1)*Nlive)

                # select a third index
                while idx3 == idx1 or idx3 == idx2 or idx3 == sampidx:
                    idx3 = ceil(numpy.random.rand(1)*Nlive)

                # select the points corresponding to the indices
                A = livepoints[idx1,:]
                B = livepoints[idx2,:]
                C = livepoints[idx3,:]

                # Define differential evolution constants
                F = 1.2     # F = 1.2 stretches the distribution of points on avg
                CR = 0.9    # Crossover probability for each dimension

                # Now iterate through the dimensions and figure out whether or
                # not we accept each change to the dimension
                sampletmp = sample
                for j in range(0,sample.shape[0]-1):
                    if numpy.random.rand(1) < CR:
                        sampletmp[j] = A[j] + F*(B[j] - C[j])
                    # else it just leaves the sample dimension alone

            # check if sample is within prior boundaries
            sampletmp = reflectbounds(sampletemp, par_range)
            newPrior = prior(sampletmp)
            # Now implement the Metropolis-Hastings rejection step, to keep
            # the random walk Markovian. This ensures that even though we use
            # differential evolution or the t distribution to generate our
            # proposal, the samples we generate are from a Markov chain whose
            # stationary distribution is still the prior distribution.

            if numpy.log(numpy.random.rand(1)) > newPrior - currentPrior: # reject point
                # Continues to next iteration of the for loop. This is akin to
                # rejecting the proposal we just made, because we left the
                # proposal as 'sampletmp' and didn't set it to sample.
                #
                continue

            # At this point, we have generated a proposal step according to the
            # t distribution or DE, and rejected it/accepted it based on the M-H
            # rule. If we stopped here, we would just generate a random walk of
            # samples from the prior.

            # We now add an additional step to reject the sample if it has
            # a likelihood lesser than a critical value. That way, repeating
            # this proposal + reject + reject will generate a random walk whose
            # distribution converges to the prior distribution conditional on
            # the likelihood being greater than a critical value.

            #
            # get the likelihood of the new sample

            logLnew = likelihood(x, data);

            # if logLnew is greater than logLmin accept point
            if logLnew > logLmin:
                acc = acc + 1
                currentPrior = newPrior
                sample = sampletmp
                logL = logLnew

        # Only break out of the while loop if at least one point is accepted,
        # otherwise try again.
        if acc > 0:
            acctot = acc
            break

        acctot = acctot + acc
        Ntimes = Ntimes + 1
        # while loop ends here.


    return (sample, logL)

def reflectbounds(new, par_range):
    # based off code by J. A. Vrugt, et al
    y = new.copy()
    minn = par_range[:,0]
    maxn = par_range[:,1]

    ny = 2*minn - y
    y = numpy.where(y > minn, y, ny)

    nyy = 2*maxn - y
    y = numpy.where(y < maxn, y, ny)

    # Now double check if all elements are within bounds.
    # If not, just pick a random point from a uniform distribution encomparssing
    # the bounds.
    ny = minn + numpy.random.rand(maxn.shape[0])*(maxn - minn)
    y = numpy.where(y > minn, y, ny)
    ny = minn + numpy.random.rand(maxn.shape[0])*(maxn - minn)
    y = numpy.where(y < maxn, y, ny)
    return y



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
