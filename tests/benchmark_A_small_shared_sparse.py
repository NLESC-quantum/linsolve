"""Benchmark a system of equations with a small number of independent parameters.

Use a large number of parallel instances that allow the inverted A matrix to be reused.
In this case, we benchmark the case when a sparse representation of A is used.
"""

import random
import time

import numpy as np

import linsolve

np.random.seed(0)

NPRMS = 10
NEQS = 100
SIZE = 1000000
# sparse: 0.82 s

prms = {"g%d" % i: np.arange(SIZE) for i in range(NPRMS)}
prm_list = list(prms.keys())

eqs = [("+".join(["%s"] * 3)) % tuple(random.sample(prm_list, 3)) for _ in range(NEQS)]

data = {eq: eval(eq, prms) for eq in eqs}

ls = linsolve.LinearSolver(data, sparse=True)
t0 = time.time()
sol = ls.solve()
t1 = time.time()

print(f"Solved in {t1 - t0}")

for k in prm_list:
    assert np.mean(np.abs(sol[k] - prms[k]) ** 2) < 1e-3
