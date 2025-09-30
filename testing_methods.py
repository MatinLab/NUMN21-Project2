# Test the optimization methods
from optMethod import Newton, GoodBroyden, BadBroyden, SymmetricBroyden, DFP, BFGS
import numpy as np

testFunction1 = lambda x: x[0]**3 + 2*x[0]*x[1] + x[1]**6 # Should have minimum at ~(0.707, -0.749)
testFunction2 = lambda x: x[0]**2 + x[1]**2 - 1  # Should have minimum at (0, 0)

newton_method = Newton()
good_broyden = GoodBroyden()
bad_broyden = BadBroyden()
symmetric_broyden = SymmetricBroyden()
dfp = DFP()
bfgs = BFGS()

x0 = np.array([5, -5], dtype=np.double)
H_inv = newton_method.approximate_hessian_inv(testFunction1, x0)
#print(H)
H = np.linalg.inv(H_inv)

# Testing exact line search
#grad = newton_method.approximate_grad(testFunction1, x0)
alpha = newton_method.exact_line_search(testFunction1, x0, newton_method.approximate_grad, H_inv)
print(f"Minimizing alpha: {alpha}")
#direction = -H_inv @ newton_method.approximate_grad(testFunction1, x0)
# Testing inexact line search
inexact_search = newton_method.create_inexact_linesearch(testFunction1, alpha_init=1)
alpha = inexact_search(testFunction1, x0, newton_method.approximate_grad, H_inv)
print(f"Inexact minimizing alpha: {alpha}")

dx = 10**-3
# Testing newton's method
print("NEWTON'S METHOD #########################")
print("Minimum for x**3 + 2xy + y**6")
minimum = newton_method.optimize(testFunction1, x0, dx=dx, termination_criterion=newton_method.residual_criterion, alpha=0.5, tol=10**-3)
print(f"Minimum of function found at {minimum} using residual stopping criterion")
minimum = newton_method.optimize(testFunction1, x0, dx=dx, termination_criterion=newton_method.cauchy_criterion, alpha=0.5, tol=10**-3)
print(f"Minimum of function found at {minimum} using Cauchy stopping criterion")

print("Minimum for x**2 + y**2 - 1")
minimum = newton_method.optimize(testFunction2, x0, dx=dx, termination_criterion=newton_method.residual_criterion, alpha=0.5, tol=10**-3)
print(f"Minimum of function found at {minimum} using residual stopping criterion")
minimum = newton_method.optimize(testFunction2, x0, dx=dx, termination_criterion=newton_method.cauchy_criterion, alpha=0.5, tol=10**-3)
print(f"Minimum of function found at {minimum} using Cauchy stopping criterion")

print("GOOD BROYDEN METHOD #########################")
print("Minimum for x**3 + 2xy + y**6")
minimum = good_broyden.optimize(testFunction1, x0, dx=dx, termination_criterion=newton_method.residual_criterion, alpha=.1, tol=10**-4)
print(f"Minimum of function found at {minimum} using residual stopping criterion")
print("Minimum for x**2 + y**2 - 1")
minimum = good_broyden.optimize(testFunction2, x0, dx=dx, termination_criterion=newton_method.residual_criterion, alpha=.1, tol=10**-4)
print(f"Minimum of function found at {minimum} using residual stopping criterion")

print("BAD BROYDEN METHOD #########################")
print("Minimum for x**3 + 2xy + y**6")
minimum = bad_broyden.optimize(testFunction1, x0, dx=dx, termination_criterion=newton_method.residual_criterion, alpha=.1, tol=10**-4)
print(f"Minimum of function found at {minimum} using residual stopping criterion")
print("Minimum for x**2 + y**2 - 1")
minimum = bad_broyden.optimize(testFunction2, x0, dx=dx, termination_criterion=newton_method.residual_criterion, alpha=.1, tol=10**-4)
print(f"Minimum of function found at {minimum} using residual stopping criterion")

print("SYMMETRIC BROYDEN METHOD #########################")
print("Minimum for x**3 + 2xy + y**6")
minimum = symmetric_broyden.optimize(testFunction1, x0, dx=dx, termination_criterion=newton_method.residual_criterion, alpha=.1, tol=10**-4)
print(f"Minimum of function found at {minimum} using residual stopping criterion")
print("Minimum for x**2 + y**2 - 1")
minimum = symmetric_broyden.optimize(testFunction2, x0, dx=dx, termination_criterion=newton_method.residual_criterion, alpha=.1, tol=10**-4)
print(f"Minimum of function found at {minimum} using residual stopping criterion")

print("DFP METHOD #########################")
print("Minimum for x**3 + 2xy + y**6")
minimum = dfp.optimize(testFunction1, x0, dx=dx, termination_criterion=newton_method.residual_criterion, alpha=.1, tol=10**-4)
print(f"Minimum of function found at {minimum} using residual stopping criterion")
print("Minimum for x**2 + y**2 - 1")
minimum = dfp.optimize(testFunction2, x0, dx=dx, termination_criterion=newton_method.residual_criterion, alpha=.1, tol=10**-4)
print(f"Minimum of function found at {minimum} using residual stopping criterion")

print("BFGS METHOD #########################")
print("Minimum for x**3 + 2xy + y**6")
minimum = bfgs.optimize(testFunction1, x0, dx=dx, termination_criterion=newton_method.residual_criterion, alpha=.1, tol=10**-4)
print(f"Minimum of function found at {minimum} using residual stopping criterion")
print("Minimum for x**2 + y**2 - 1")
minimum = bfgs.optimize(testFunction2, x0, dx=dx, termination_criterion=newton_method.residual_criterion, alpha=.1, tol=10**-4)
print(f"Minimum of function found at {minimum} using residual stopping criterion")


# Testing BFGS with inexact line search
print("INEXACT LINE SEARCH ###################")
print("Minimum for x**3 + 2xy + y**6")
minimum = bfgs.optimize(testFunction1, x0, dx=dx, termination_criterion=newton_method.cauchy_criterion, line_search=inexact_search, tol=10**-4)
print(f"Minimum of function found at {minimum} using residual stopping criterion and inexact line search")