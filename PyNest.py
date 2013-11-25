# PyNest - a Python implementation of nested sampling, based on matlab multinest
# originally implemented by Matthew Pitkin et al. This code mostly just ports
# Pitkin's code directly, with a few modifications to the DE code. Right now,
# the multinest based sampling actually is not implemented -- the only choice
# is to evolve the samples in the prior distribution by the use of MCMC, with
# a Student's t distribution and differential-evolution-based proposal
# mechanism.
#
# Port started by David Christle <christle@uchicago.edu>, Nov. 2013

import numpy

def nested_sampler(data, Nlive, Nmcmc, tolerance, likelihood,
    model, prior, priordraw, extraparams, D):
    # Nmcmc cannot be zero, for this initial code, since we always use MCMC
    # sampling and not the ellipse-based sampling, for the initial port of this
    # code from Matlab Multinest. This is just simpler for now and faster to
    # write if we omit features.

    # Get the number of parameters from the prior array
    D = prior.shape[0]

    # Skip getting parameter names... not sure if that's useful for this code.

    # Draw the set of initial live points from the prior

    # This priordraw function is new versus the Pitkin implementation; trying
    # to keep this general, as long as the user can write a function that
    # does the job of sampling from the prior.

    livepoints = priordraw(Nlive)

    # calculate the log likelihood of all the live points
    logL = numpy.zeros(Nlive)
    for i in range(0,Nlive-1):
        logL[i] = likelihood(Nlive[i])

    # don't scale parameters - don't see the reason quite yet

    # initial tolerance
    tol = numpy.inf

    # initial width of prior volume (from X_0 = 1 to X_1 = exp(-1/N))
    logw = numpy.log(1 - numpy.exp(-1/Nlive))

    # initial log evidence (Z=0)
    logZ = -numpy.inf

    # initial information
    H = 0

    # initialize array of samples for posterior
    nest_samples = numpy.zeros((1,D+1),float)

    # some initial values if MCMC nested sampling is used
    # value to scale down the covariance matrix -- can change if required
    propscale = 0.1

    # some initial values if multinest sampling is used -- not currently
    # implemented
    h = 1.1 # h values from bottom of p. 1605 of Feroz and Hobson
    FS = h # start FS at h, so ellipsoidal partitioning is done first time
    K = 1 # start with one cluster of live points

    # get maximum likleihood
    logLmax = numpy.max(logL)

    # initialize iteration counter
    j = 1

    # MAIN LOOP
    while tol > tolerance or j <= Nlive:

        # expected value of true remaining prior volume X
        VS = numpy.exp(-j/Nlive)

        # find maximum of likelihoods

        logLmin = numpy.min(logL)
        min_idx = numpy.argmin(logL)

        # set the sample to the minimum value
        nest_samples[j,:] = numpy.concatenate(livepoints[min_idx,:], logLmin)

        # get the log weight (Wt = L*w)
        logWt = logLmin + logw

        # save old evidence and information
        logZold = logZ.copy()
        Hold = H.copy()

        # update evidence, information, and width
        logZ = logplus(logZ, logWt)
        H = numpy.exp(logWt - logZ)*logLmin + \
            numpy.exp(logZold - logZ)*(Hold + logZold) - logZ
        # logw = logw - logt(Nlive) -- this comment leftover from Pitkin
        logw = logw - 1/Nlive

        if Nmcmc > 0:
            # do MCMC nested sampling -- this is the only option implemented
            # so far.

            # get the Cholesky decomposed covariance of the live points
            # (do every 100th iteration, can change this if required)
            if j-1 % 100 == 0:
                # NOTE that for numbers of parameters >~10 covariances are often
                # not positive definite and cholcov will have "problems".
                # cholmat = cholcov(propscale*cov(livepoints) -- original code
                cholmat = numpy.linalg.cholesky(propscale*numpy.cov(livepoints))
                # use modified Cholesky decomposition, which works even for
                # matrices that are not quite positive definite from:
                # http://infohost.nmt.edu/~borchers/ldlt.html
                # (via http://stats.stackexchange.com/questions/6364
                # /making-square-root-of-covariance-matrix-positive-definite-matlab
                #cv = numpy.cov(livepoints)
                #l, d = mchol(propscape*cv)

                # cholmat = numpy.transpose(l)*matrix square root needed here(d)

        # draw a new sample using mcmc algorithm
            point_draw, logL_draw = draw_mcmc(livepoints, cholmat, logLmin, \
                prior, prior_bounds, data, likelihood, Nmcmc)
            livepoints[min_idx,:] = point_draw.copy()
            logL[min_idx] = logL_draw.copy()

        else:
            print 'Nmcmc is not > 0 -- multinest sampling not implemented yet!'

        # Now update maximum likelihood if appropriate
        if logL(min_idx) > logLmax:
            logLmax = logL[min_idx].copy()


        # Work out tolerance for stopping criterion
        tol = logplus(logZ, logLmax - (j/Nlive)) - logZ

        # Display progress
        print 'log(Z): %.5e, tol = %.5e, K = %d, iteration = %d' % (logZ, tol,
            K, j)
        # update counter
        j = j+1
    # Sort the remaining points (in order of likelihood) and add them on to the
    # evidence
    logL_sorted_args = numpy.argsort(logL)
    livepoints_sorted = livepoints[logL_sorted_args,:]

    for i in range(0,Nlive-1):
        logZ = logplus(logZ, logL_sorted[i] + logw)

    # append the additional livepoints to the nested samples
    nest_samples = numpy.concatenate(nest_samples,numpy.concatenate(livepoints_sorted, logL_sorted))
    # really not sure if the below indexing even works correctly
    post_samples = nest_samples[numpy.nonzero(logWt > logrand)]

    return (logZ, nest_samples, post_samples)

def nest2pos(nest_samples, Nlive):
    # Code originally by John Veitch (2009) and J. Romano (2012)
    N = nest_samples.shape[0]
    Ncol = nest_samples.shape[1]

    # calulcate logWt = log(L*w) = logL + logw = logL - i/Nlive
    logL = nest_samples[:,Ncol-1]
    logw = -numpy.append(numpy.array(numpy.transpose(numpy.array(range(1,N-Nlive)))),
        (N-Nlive)*numpy.ones((Nlive,1))/Nlive)
    logWt = logL + logw

    # posterior samples are given by the normalized weight
    logWtmax = numpy.max(logWt)
    logWt = logWt - logWtmax # Wt -> Wt/Wmax

    # accept a nested sample as a posterior only if its value is > than a random
    # number drawn from a uniform distribution
    logrand = numpy.log(numpy.random.rand((N,1)))


def draw_mcmc(livepoints, cholmat, logLmin,
    prior, prior_bounds, data, likelihood, Nmcmc):

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
