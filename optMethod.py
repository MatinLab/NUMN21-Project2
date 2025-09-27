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
    # Numerically approximate the gradient of a function
    def approximate_grad(self, f, x, dx=0.01):
        # Approximate the gradient of a function f about the vector x
        # x must be a numpy array and f must take numpy arrays as arguments
        # dx is the delta to add to each variable
        grad = np.zeros(x.shape)
        f0 = f(x)
        for i in range(len(grad)):
            x1 = x.copy()
            x1[i] += dx
            grad[i] = (f(x1)-f0)/dx
        return grad
    # Function to approximate the Hessian of a function
    def approximate_hessian(self, f, x, dx=0.000001):
        H = np.zeros((len(x), len(x)))
        grad0 = self.approximate_grad(f, x, dx)
        for i in range(len(x)):
            # For each xi, we get the derivative of the gradient with
            # respect to the xi
            # This is the ith row in the Hessian
            x1 = x.copy()
            x1[i] += dx
            H[i] = (self.approximate_grad(f, x1, dx)-grad0)/dx
        return H
    
    
class Newton(OptMethod):
    # Does Newton method
    def optimize(self, function, gradient=None):
        # Run the optimization on the given function with optionally the given gradient
        pass
