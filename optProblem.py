import numpy as np
from optMethod import *

# Optimization class
class OptProblem():
    
    # Constructor
    def __init__(self, function, gradient=None):
        self.function = function
        self.gradient = gradient
        self.OptMethod = OptMethod()
    
    def newton(self, x0, dx=10**-4, termination_criterion=None, tol=10**-4, alpha=None, line_search=None, max_iterations=10**5):
        # Do newton method on the function
        newtonMethod = Newton()
        result = newtonMethod.optimize(self.function, x0, self.gradient, dx, termination_criterion, tol, alpha, line_search, max_iterations)
        return result
    def good_broyden(self, x0, dx=10**-4, termination_criterion=None, tol=10**-4, alpha=None, line_search=None, max_iterations=10**5):
        # Do good broyden method on the function
        gBroydenMethod = GoodBroyden()
        result = gBroydenMethod.optimize(self.function, x0, self.gradient, dx, termination_criterion, tol, alpha, line_search, max_iterations)
        return result
    def bad_broyden(self, x0, dx=10**-4, termination_criterion=None, tol=10**-4, alpha=None, line_search=None, max_iterations=10**5):
        # Do bad broyden method on the function
        bBroydenMethod = BadBroyden()
        result = bBroydenMethod.optimize(self.function, x0, self.gradient, dx, termination_criterion, tol, alpha, line_search, max_iterations)
        return result
    def symmetric_broyden(self, x0, dx=10**-4, termination_criterion=None, tol=10**-4, alpha=None, line_search=None, max_iterations=10**5):
        # Do symmetric broyden method on the function
        symBroydenMethod = SymmetricBroyden()
        result = symBroydenMethod.optimize(self.function, x0, self.gradient, dx, termination_criterion, tol, alpha, line_search, max_iterations)
        return result
    def dfp(self, x0, dx=10**-4, termination_criterion=None, tol=10**-4, alpha=None, line_search=None, max_iterations=10**5):
        # Do dfp method on the function
        dfpMethod = DFP()
        result = dfpMethod.optimize(self.function, x0, self.gradient, dx, termination_criterion, tol, alpha, line_search, max_iterations)
        return result
    def bfgs(self, x0, dx=10**-4, termination_criterion=None, tol=10**-4, alpha=None, line_search=None, max_iterations=10**5):
        # Do bfgs method on the function
        bfgsMethod = BFGS()
        result = bfgsMethod.optimize(self.function, x0, self.gradient, dx, termination_criterion, tol, alpha, line_search, max_iterations)
        return result
    def bfgs_demo(self, x0, dx=10**-4, termination_criterion=None, tol=10**-4, alpha=None, line_search=None, max_iterations=10**5):
        # Do bfgs method on the function
        bfgsMethod = BFGS_demo()
        result = bfgsMethod.optimize(self.function, x0, self.gradient, dx, termination_criterion, tol, alpha, line_search, max_iterations)
        return result, bfgsMethod.Hks, bfgsMethod.points

'''Usage:        
myfunc = None
mygrad = None
problem = OptProblem(myfunc, mygrad)
results = problem.newton(params)'''