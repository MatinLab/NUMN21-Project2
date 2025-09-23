import numpy as np

class OptMethod():
    # Generic class for Quasi Newton methods
    # Constructor
    def __init__(self):
        pass
    # Optimize method
    def optimize(self, function, gradient=None):
        # To be overridden in the child class
        pass
    
    
class Newton(OptMethod):
    # Does Newton method
    def optimize(self, function, gradient=None):
        # Run the optimization on the given function with optionally the given gradient
        pass
