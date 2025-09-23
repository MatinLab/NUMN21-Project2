import numpy as np
from optMethod import *

# Optimization class
class OptProblem():
    
    # Constructor
    def __init__(self, obj_function, obj_gradient=None):
        self.obj_function = obj_function
        self.obj_gradient = obj_gradient
    
    def newton(self):
        # Do newton method on the function
        newtonMethod = Newton()
        result = newtonMethod.optimize(self.obj_function, self.obj_gradient)
        return result
        
myfunc = None
mygrad = None
problem = OptProblem(myfunc, obj_gradient=mygrad)
results = problem.newton()