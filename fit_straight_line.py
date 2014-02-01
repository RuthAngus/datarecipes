import numpy as np
import matplotlib.pyplot as pl

def fit_line(x, y, yerr):
    Y = (np.matrix(y)).T
    A = np.matrix((np.ones(len(x)), x)).T
    C = yerr**2 * np.identity(len(x))
    return np.array((np.linalg.inv(A.T * np.linalg.inv(C) * A)) * (A.T * np.linalg.inv(C) * Y)),\
      np.sqrt((np.array((np.linalg.inv(A.T * np.linalg.inv(C) * A)))[0][0])), \
      np.sqrt((np.array((np.linalg.inv(A.T * np.linalg.inv(C) * A)))[1][1]))

def chi_squared(x, y, yerr):
    Y = (np.matrix(y)).T
    A = np.matrix((np.ones(len(x)), x)).T
    C = yerr**2 * np.identity(len(x))
    X = (np.linalg.inv(A.T * np.linalg.inv(C) * A)) * (A.T * np.linalg.inv(C) * Y)
    return np.array((Y - A * X).T * np.linalg.inv(C) * (Y - A * X))[0]

def fit_quad(x, y, yerr):
    Y = (np.matrix(y)).T
    A = np.matrix((np.ones(len(x)), x, x**2)).T
    C = yerr**2 * np.identity(len(x))
    return np.array((np.linalg.inv(A.T * np.linalg.inv(C) * A)) * (A.T * np.linalg.inv(C) * Y)),\
      np.sqrt((np.array((np.linalg.inv(A.T * np.linalg.inv(C) * A)))[0][0])), \
      np.sqrt((np.array((np.linalg.inv(A.T * np.linalg.inv(C) * A)))[1][1])), \
      np.sqrt((np.array((np.linalg.inv(A.T * np.linalg.inv(C) * A)))[2][2]))

data = np.genfromtxt('/Users/angusr/Python/datarecipes/data.txt').T
x = data[0]
y = data[1]
yerr = data[2]
xerr = data[3]
pxy = data[4]

# plot data
pl.clf()
pl.errorbar(x, y, xerr = xerr, yerr = yerr, fmt = 'k.')
pl.savefig('all_data')
pl.clf()
pl.errorbar(x[5:], y[5:], xerr = xerr[5:], yerr = yerr[5:], fmt = 'k.')
pl.savefig('data')


params, b_uncert, m_uncert = fit_line(x, y, yerr)
b = params[0][0]
m = params[1][0]
print 'b = ', b, '+/-', b_uncert
print 'm = ', m, '+/-', m_uncert

pl.clf()
pl.errorbar(x, y, yerr = yerr, fmt = 'k.')
pl.plot(x, (m*x + b), 'k-')
pl.xlabel('x')
pl.ylabel('y')
pl.show()

params, b_uncert, m_uncert, q_uncert = fit_quad(x[5:], y[5:], yerr[5:])
print params
b = params[0][0]
m = params[1][0]
q = params[2][0]
print 'b = ', b, '+/-', b_uncert
print 'm = ', m, '+/-', m_uncert
print 'q = ', q, '+/-', q_uncert

pl.clf()
pl.errorbar(x[5:], y[5:], yerr = yerr[5:], fmt = 'k.')
pl.plot(x, (q*x**2 + m*x + b), 'ro')
pl.xlabel('x')
pl.ylabel('y')
pl.show()
