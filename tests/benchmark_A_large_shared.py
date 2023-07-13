'''Benchmark a system of equations with a large number of independent
parameters and a modest number of parallel instances that allow the
inverted A matrix to be reused.'''
import linsolve
import numpy as np
import time, random

np.random.seed(0)

NPRMS = 2000
NEQS = 5000
SIZE = 100
sparse = False # sparse: 1.30 s, dense: 1.50 s

prms = {'g%d' % i: np.arange(SIZE) for i in range(NPRMS)}
prm_list = list(prms.keys())

eqs = [('+'.join(['%s'] * 5))  % tuple(random.sample(prm_list, 5)) 
            for i in range(NEQS)]

data = {eq: eval(eq, prms) for eq in eqs}

ls = linsolve.LinearSolver(data, sparse=sparse)
t0 = time.time()
sol = ls.solve()
t1 = time.time()

print('Solved in {}'.format(t1-t0))

for k in prm_list:
    assert np.mean(np.abs(sol[k] - prms[k])**2) < 1e-3
