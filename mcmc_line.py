import numpy as np
import emcee
import scipy.optimize as op
import matplotlib.pyplot as pl

def lnlike(theta, x, y, yerr):
    m, b = theta
    model = m * x + b
    inv_sigma2 = 1.0/(yerr**2)
    return -0.5*(np.sum((y-model)**2*inv_sigma2 - np.log(inv_sigma2)))

def lnprior(theta):
    m, b = theta
    if 0.0 < m < 5 and 0.0 < b < 100.0:
        return 0.0
    return -np.inf

def lnprob(theta, x, y, yerr):
    lp = lnprior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike(theta, x, y, yerr)

data = np.genfromtxt('/Users/angusr/Python/datarecipes/data.txt').T
x = data[0][5:]
y = data[1][5:]
yerr = data[2][5:]
xerr = data[3][5:]
pxy = data[4][5:]

m_init = 2.
b_init = 34.

pl.close(1)
pl.figure(1)
pl.errorbar(x, y, yerr = yerr, fmt = 'k.')
pl.show()

# Calculate maximum-likelihood values
nll = lambda *args: -lnlike(*args)
result = op.fmin(nll, [m_init, b_init], args=(x, y, yerr))
m_ml, b_ml = result
print result

# Run emcee
ndim, nwalkers, nsteps = 2, 100, 2000
pos = [result + 1e-4*np.random.randn(ndim) for i in range(nwalkers)]
sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(x, y, yerr))
sampler.run_mcmc(pos, nsteps)

pl.close(2)
pl.figure(2)
pl.subplot(2,1,1)
[pl.plot(range(nsteps), sampler.chain[i, :, 0], 'k-', alpha = 0.2) for i in range(100)]
pl.plot(range(nsteps), np.ones(nsteps)*m_init, 'r--')
pl.subplot(2,1,2)
[pl.plot(range(nsteps), sampler.chain[i, :, 1], 'k-', alpha = 0.2) for i in range(100)]
pl.plot(range(nsteps), np.ones(nsteps)*b_init, 'r--')
pl.show()

# Flatten chain
samples = sampler.chain[:, 50:, :].reshape((-1, ndim))

import triangle
fig = triangle.corner(samples, labels=["$m$", "$b$", "$\ln\,f$"])
fig.savefig("triangle.png")

