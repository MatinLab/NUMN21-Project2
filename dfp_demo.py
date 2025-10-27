import numpy as np
import matplotlib.pyplot as plt
from optProblem import OptProblem

# test function
def myFunction(x):
    return x[0]**2 + x[1]**2

solver = OptProblem(myFunction)

# Use the demo dfp to show the decorator working
termination_condition = solver.OptMethod.residual_criterion
inexact_search = solver.OptMethod.create_inexact_linesearch(solver.function, alpha_init=1)
x0 = np.array([-5, 2.])
minimum, Hk_invs, points = solver.dfp_demo(x0, termination_criterion=termination_condition, alpha=0.5)
print(f"The last 3 points that were checked:\n{points[-3:]}")
print(f"The last hessian matrix:\n{np.linalg.inv(Hk_invs[-1])}")

