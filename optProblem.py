import numpy as np
from optMethod import *

# Optimization class
class OptProblem():
    
    # Constructor
    def __init__(self, function, gradient=None):
        self.function = function
        self.gradient = gradient
        self.OptMethod = OptMethod()

    def _run_method(self, method_class, x0, dx=1e-4, termination_criterion=None,
                    tol=1e-4, alpha=None, line_search=None, max_iterations=1e5):
        # Execute a generic optimization method
        method = method_class()
        return method.optimize(
            self.function, x0, self.gradient, dx,
            termination_criterion, tol, alpha,
            line_search, max_iterations
        )
    
    def newton(self, *args, **kwargs):
        # Do newton method on the function
        return self._run_method(Newton, *args, **kwargs)
    def good_broyden(self, *args, **kwargs):
        # Do good broyden method on the function
        return self._run_method(GoodBroyden, *args, **kwargs)
    def bad_broyden(self, *args, **kwargs):
        # Do bad broyden method on the function
        return self._run_method(BadBroyden, *args, **kwargs)
    def symmetric_broyden(self, *args, **kwargs):
        # Do symmetric broyden method on the function
        return self._run_method(SymmetricBroyden, *args, **kwargs)
    def dfp(self, *args, **kwargs):
        # Do dfp method on the function
        return self._run_method(DFP, *args, **kwargs)
    def bfgs_demo(self, x0, dx=10**-4, termination_criterion=None, tol=10**-4, alpha=None, line_search=None, max_iterations=10**5):
        # Do bfgs method on the function
        bfgsMethod = BFGS_demo()
        result = bfgsMethod.optimize(self.function, x0, self.gradient, dx, termination_criterion, tol, alpha, line_search, max_iterations)
        return result, bfgsMethod.Hks, bfgsMethod.points
    
    def dfp_demo(self, x0, dx=10**-4, termination_criterion=None, tol=10**-4, alpha=None, line_search=None, max_iterations=10**5):
        # Do dfp method on the function
        dfpMethod = DFP_demo()
        result = dfpMethod.optimize(self.function, x0, self.gradient, dx, termination_criterion, tol, alpha, line_search, max_iterations)
        return result, dfpMethod.Hks, dfpMethod.points

'''Usage:        
myfunc = None
mygrad = None
problem = OptProblem(myfunc, mygrad)
results = problem.newton(params)'''