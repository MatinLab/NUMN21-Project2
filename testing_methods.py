# Test the optimization methods
from optMethod import Newton
import numpy as np

testFunction1 = lambda x: x[0]**3 + 2*x[0]*x[1] + x[1]**6
testFunction2 = lambda x: x[0]**2 + x[1]**2 - 1  # Should have minimum at (0, 0)

newton_method = Newton()

x0 = np.array([1, -1.])
H = newton_method.approximate_hessian(testFunction1, x0)
#print(H)
H_inv = np.linalg.inv(H)

# Testing exact line search
grad = newton_method.approximate_grad(testFunction1, x0)
alpha = newton_method.exact_line_search(testFunction1, x0, grad, H_inv)
print(f"Minimizing alpha: {alpha}")

dx = 10**-4
print("Minimum for x**3 + 2xy + y**6")
minimum = newton_method.optimize(testFunction1, x0, dx=dx, termination_criterion=newton_method.residual_criterion, tol=10**-3)
print(f"Minimum of function found at {minimum} using residual stopping criterion")
minimum = newton_method.optimize(testFunction1, x0, dx=dx, termination_criterion=newton_method.cauchy_criterion, tol=10**-3)
print(f"Minimum of function found at {minimum} using Cauchy stopping criterion")

print("Minimum for x**2 + y**2 - 1")
minimum = newton_method.optimize(testFunction2, x0, dx=dx, termination_criterion=newton_method.residual_criterion, tol=10**-3)
print(f"Minimum of function found at {minimum} using residual stopping criterion")
minimum = newton_method.optimize(testFunction2, x0, dx=dx, termination_criterion=newton_method.cauchy_criterion, tol=10**-3)
print(f"Minimum of function found at {minimum} using Cauchy stopping criterion")



