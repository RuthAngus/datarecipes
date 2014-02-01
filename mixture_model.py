import numpy as np
import emcee
import scipy.optimize as op
import matplotlib.pyplot as pl
import triangle

def line(theta, x):
    return theta[0]*x + theta[1]

def lnlike(theta, x, y, yerr):
    m, b, Y, V, P = theta
    model1 = line([m,b], x)
    model2 = Y
    inv_sigma21 = 1./(yerr**2)
    inv_sigma22 = 1./(V + yerr**2)
    return -0.5*(np.sum((y-model1)**2*inv_sigma21 - np.log((1.-P)*inv_sigma21)))\
            -0.5*(np.sum((y-model2)**2*inv_sigma22 - np.log(P*inv_sigma22)))

def lnprior(theta):
    m, b, Y, V, P = theta
    if 0 < m < 5 and -100 < b < 400 and 0 < Y < 700 and 0 < V < 10000 and 0 < P < 1:
        return 0.0
    return -np.inf

def lnprob(theta, x, y, yerr):
    lp = lnprior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike(theta, x, y, yerr)

data = np.genfromtxt('/Users/angusr/Python/datarecipes/data.txt').T
x = data[0]#[5:]
y = data[1]#[5:]
yerr = data[2]#[5:]
xerr = data[3]#[5:]
pxy = data[4]#[5:]

# m, b, Yb, Vb, Pb
theta_init = [2., 34., 400., 200., .5]

# Calculate maximum-likelihood values
nll = lambda *args: -lnlike(*args)
result = op.fmin(nll, theta_init, args=(x, y, yerr))
print result

# Run emcee
ndim, nwalkers = 5, 32
pos = [theta_init + 1e-2*np.random.randn(ndim) for i in range(nwalkers)]
sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(x, y, yerr))
print("Burn-in")
pos, lp, state = sampler.run_mcmc(pos, 100)
sampler.reset()
print("Full Run")
sampler.run_mcmc(pos, 500)

# plot traces
pl.figure()
for i in range(ndim):
    pl.clf()
    pl.plot(sampler.chain[:, :, i].T)
    pl.savefig("{0}.png".format(i))

# Flatten chain
samples = sampler.chain[:, 50:, :].reshape((-1, ndim))

# Make triangle plot
fig_labels = ["$m$", "$b$", "$Y_b$", "$V_b$", "$P_b$"]
fig = triangle.corner(samples, labels=fig_labels)
fig.savefig("triangle.png")

# show the covariance of m and b only.
mb_samples = samples[:, :2]
fig = triangle.corner(mb_samples, labels=fig_labels[:2])
fig.savefig("m_b")

# Find values
mcmc_result = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),
                             zip(*np.percentile(samples, [16, 50, 84],
                                                axis=0)))
mcmc_result =  np.array(mcmc_result)[:,0]
print 'mcmc result', mcmc_result

# plot draws from the distribution
pl.clf()
pl.errorbar(x, y, yerr=yerr, fmt = 'k.')
# lims = pl.xlim(pl.gca().get_xlim())
xp = np.arange(0, 300, 1.)
for i in range(20):
    m = mb_samples[:, 0][np.random.uniform(0, len(mb_samples[:, 0]))]
    b = mb_samples[:, 1][np.random.uniform(0, len(mb_samples[:, 1]))]
    pl.plot(xp, (m*xp+b), 'k-', alpha=0.2)
pl.plot(xp, mcmc_result[0]*xp+mcmc_result[1], "r-", alpha=0.5)
pl.plot(xp, result[0]*xp+result[1], "b-", alpha=0.5)
pl.xlim(0, 300)
pl.ylim(0, 700)
pl.savefig("draws")
