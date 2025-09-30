# Use the optimization problem class to run bfgs on the chebychev eq
import numpy as np
import scipy.optimize as so
from optProblem import OptProblem
from chebyquad_problem import *

# Wrap the gradient in the expected format for our functions
cheby_grad = lambda f, x, dx: gradchebyquad(x)
solver = OptProblem(chebyquad, cheby_grad)

# X points, 4 8 and 11 values between 0 and 1
termination_condition = solver.OptMethod.cauchy_criterion
alpha = 0.1
for i in [4, 8, 11]:
    print(f"Approximating using {i} points")
    x = linspace(0, 1, i)
    # Our optimizer
    ourmin = solver.bfgs(x, termination_criterion=termination_condition, alpha=alpha)
    print(f"Minimum output from our solver:\n\t{ourmin}")
    # Scipy output
    xmin = so.fmin_bfgs(chebyquad, x, gradchebyquad, disp=False)
    print(f"Minimum output from scipy:\n\t{xmin}")
    print()