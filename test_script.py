# PyNest testing script -- the purpose of this is just to see if it works!

import PyNest
import numpy

def main():
    prior_bnds = numpy.array([[0,1],[0,1]])
    amp = 10.0
    phi = 2.3
    f = 24.788634

    dt = 1.0/100.0
    tlen = 3.0
    t = numpy.arange(0,tlen+dt,dt)

    y = amp*numpy.sin(2*numpy.pi*f*t + phi)
    data = numpy.column_stack((t,y))
    print 'data shape %d' % data.shape[1]
    tolerance = 0.1


    Nlive = 500
    Nmcmc = 20
    maxIter = 5000
    PyNest.nested_sampler(data, Nlive, maxIter, Nmcmc, tolerance, sin_likelihood, sin_prior_function, sin_pr_draw, prior_bnds, 2)
def sin_pr_draw(Npoints,D):
    output = numpy.random.rand(Npoints,2)
    return output

def sin_prior_function(x):
    return 1/(2*numpy.pi-0)*1/(20-0)

def sin_likelihood(x, data):
    ll = 0

    for i in range(0,int(data.shape[0])-1):

        sin_model = x[0]*numpy.sin(2*numpy.pi*24.788634*data[i,0] + x[1])
        ll = ll + numpy.log(1/numpy.sqrt(2*numpy.pi*2)*numpy.exp(-numpy.power((data[i,1]-sin_model),2)*0.5/2.0))
    return ll

