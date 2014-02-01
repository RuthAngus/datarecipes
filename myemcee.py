import numpy as np
import matplotlib.pyplot as pl

# Set true parameter values
m_true = -0.9594
b_true = 4.294
f_true = 0.534

# Generate data
N = 50
x = np.sort(10 * np.random.rand(N))
yerr = 0.1 + 0.5 * np.random.rand(N)
y = m_true * x + b_true
y += np.abs(f_true * y) * np.random.randn(N)
y += yerr * np.random.randn(N)

# Calculate least-square values
A = np.vstack((np.ones_like(x), x)).T
C = np.diag(yerr * yerr)
cov = np.linalg.inv(np.dot(A.T, np.linalg.solve(C, A)))
b_ls, m_ls = np.dot(cov, np.dot(A.T, np.linalg.solve(C, y)))

# Likelihood
def lnlike(theta, x, y, yerr):
    m, b, lnf = theta
    model = m * x + b
    inv_sigma2 = 1.0/(yerr**2 + model**2*np.exp(2*lnf))
    return -0.5*(np.sum((y-model)**2*inv_sigma2 - np.log(inv_sigma2)))

# Calculate maximum-likelihood values
import scipy.optimize as op
nll = lambda *args: -lnlike(*args)
result = op.fmin(nll, [m_true, b_true, np.log(f_true)], args=(x, y, yerr))
m_ml, b_ml, lnf_ml = result

# Priors
def lnprior(theta):
    m, b, lnf = theta
    if -5.0 < m < 0.5 and 0.0 < b < 10.0 and -10.0 < lnf < 1.0:
        return 0.0
    return -np.inf

# Posterior probability dist
def lnprob(theta, x, y, yerr):
    lp = lnprior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike(theta, x, y, yerr)

pl.close(1)
pl.figure(1)
pl.errorbar(x, y, yerr = yerr, fmt = 'k.')
pl.plot(x, m_true*x+b_true, 'k-')
pl.plot(x, m_ml*x+b_ml, 'r-')
pl.plot(x, m_ls*x+b_ls, 'k--')
pl.show()

ndim, nwalkers = 3, 100
pos = [result + 1e-4*np.random.randn(ndim) for i in range(nwalkers)]

import emcee
sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(x, y, yerr))

sampler.run_mcmc(pos, 500)

pl.close(2)
pl.figure(2)
pl.subplot(3,1,1)
[pl.plot(range(500), sampler.chain[i, :, 0], 'k-', alpha = 0.2) for i in range(100)]
pl.plot(range(500), np.ones(500)*m_true, 'r-')
pl.subplot(3,1,2)
[pl.plot(range(500), sampler.chain[i, :, 1], 'k-', alpha = 0.2) for i in range(100)]
pl.plot(range(500), np.ones(500)*b_true, 'r-')
pl.subplot(3,1,3)
[pl.plot(range(500), np.exp(sampler.chain[i, :, 2]), 'k-', alpha = 0.2) for i in range(100)]
pl.plot(range(500), np.ones(500)*f_true, 'r-')
pl.show()

# Flatten chain
samples = sampler.chain[:, 50:, :].reshape((-1, ndim))

import triangle
fig = triangle.corner(samples, labels=["$m$", "$b$", "$\ln\,f$"],
                      truths=[m_true, b_true, np.log(f_true)])
fig.savefig("triangle.png")

import matplotlib.pyplot as pl
xl = np.arange(0, 10, 0.01)
for m, b, lnf in samples[np.random.randint(len(samples), size=100)]:
    pl.plot(xl, m*xl+b, color="k", alpha=0.1)
pl.plot(xl, m_true*xl+b_true, color="r", lw=2, alpha=0.8)
pl.errorbar(x, y, yerr=yerr, fmt=".k")
pl.show()

samples[:, 2] = np.exp(samples[:, 2])
m_mcmc, b_mcmc, f_mcmc = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),
                             zip(*np.percentile(samples, [16, 50, 84],
                                                axis=0)))

print m_mcmc, b_mcmc, f_mcmc
