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
    def bfgs(self, *args, **kwargs):
        # Do bfgs method on the function
        return self._run_method(BFGS, *args, **kwargs)
    def bfgs_demo(self, *args, **kwargs):
        # Do bfgs method on the function
        return self._run_method(BFGS_demo, *args, **kwargs)
    # Another demo function to show it can be used in general on other solvers
    def dfp_demo(self, *args, **kwargs):
        # Do dfp method on the function
        return self._run_method(DFP_demo, *args, **kwargs)

'''Usage:        
myfunc = None
mygrad = None
problem = OptProblem(myfunc, mygrad)
results = problem.newton(params)'''