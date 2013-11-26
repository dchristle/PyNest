# PyNest testing script -- the purpose of this is just to see if it works!
# David Christle <christle@uchicago.edu>, November 2013

import PyNest
import numpy
from matplotlib.colors import LogNorm
from pylab import *

#normal distribution center at x=0 and y=5
x = randn(100000)
y = randn(100000)+5


def main():
    prior_bnds = numpy.array([[0,20],[0,2*numpy.pi]])
    amp = 10.0
    phi = 2.3
    f = 24.788634

    dt = 1.0/100.0
    tlen = 3.0
    t = numpy.arange(0,tlen+dt,dt)
    sigma2 = 2
    noise = numpy.random.randn(t.shape[0])*numpy.sqrt(sigma2)
    y = amp*numpy.sin(2*numpy.pi*f*t + phi) + noise
    data = numpy.column_stack((t,y))
    print 'data shape %d' % data.shape[1]
    tolerance = 1


    Nlive = 500
    Nmcmc = 25
    maxIter = 10000
    logZ, nest_samples, post_samples = PyNest.nested_sampler(data, Nlive,
        maxIter, Nmcmc, tolerance, sin_likelihood, sin_prior_function,
        sin_pr_draw, prior_bnds, 2)
    hist2d(post_samples[:,0], post_samples[:,1], bins=40)
def sin_pr_draw(Npoints,D):
    output = numpy.random.rand(Npoints,2)
    return output

def sin_prior_function(x):
    return 1/(2*numpy.pi-0)*1/(20-0)

def sin_likelihood(x, data):
    sin_model = x[0]*numpy.sin(2*numpy.pi*24.788634*data[:,0] + x[1])
    ll_ind = numpy.log(1/numpy.sqrt(2*numpy.pi*2)*numpy.exp(-numpy.power((data[:,1]-sin_model),2)*0.5/2.0))
    return numpy.sum(ll_ind)

