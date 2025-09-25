import unittest

import numpy as np
from optProblem import OptProblem
from optMethod import Newton

def rosen(x):
    x = np.asarray(x, dtype=float)
    return 100.0*(x[1]-x[0]**2)**2 + (1.0 - x[0])**2

prob = OptProblem(rosen)
solver = Newton(tol=1e-8, max_iter=200, use_line_search=True)

x0 = np.array([-1.2, 1.0])
xmin, fmin, info = solver.optimize(prob, x0=x0)
print("x* =", xmin)
print("f* =", fmin)
print("status:", info["status"], "iters:", info["iters"], "||g||:", info["grad_norm"])
