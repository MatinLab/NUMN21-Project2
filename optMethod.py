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
    def approximate_grad(self, f, x, dx=10**-7):
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
    def approximate_hessian(self, f, x, dx=10**-7):
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
    # STOPPING CRITERIA
    # A stopping criterion function should take the current point,
    # the next point, the gradient, the inverse Hessian, and the tolerance.
    # Should return True or False depending on whether the method should stop
    # I.e. take the values from x1=x0 + s, where s=-H^-1*grad
    # Residual stopping criterion
    @staticmethod
    def residual_criterion(x0, x1, grad, H_inv, tol=10**-4):
        # Determine whether the residual is small enough
        if np.linalg.norm(grad) < tol:
            return True
        return False
    # Cauchy criterion
    @staticmethod
    def cauchy_criterion(x0, x1, grad, H_inv, tol=10**-4):
        # Determine whether the correction is small enough
        if np.linalg.norm(x1-x0) < tol:
            return True
        return False
        
class Newton(OptMethod):
    # Newtons method for finding the root of the gradient
    def optimize(self, function, x0, gradient=None, dx=10**-7, termination_criterion=None, tol=10**-4, max_iterations=10**5):
        # Run the optimization on the given function with optionally the given gradient
        # Returns the point at which the gradient was found to be zero
        # Default to residual if no criterion provided
        if termination_criterion is None:
            termination_criterion = self.residual_criterion
        # Default to numerical gradient if no gradient provided
        if gradient is None:
            gradient = self.approximate_grad
        # Run Newton's method
        H_inv = np.linalg.inv(self.approximate_hessian(function, x0, dx))
        grad = gradient(function, x0, dx)
        Sk = -1*H_inv @ grad
        x1 = x0 + Sk
        i = 0
        while not termination_criterion(x0, x1, grad, H_inv, tol):
            # Hasn't converged, make another step
            x0 = x1.copy()
            #try:
                #H_inv = np.linalg.inv(self.approximate_hessian(function, x0, dx))
            #except np.linalg.LinAlgError:
                #print("LINALGERROR")
                #rint(x0)
                #print(self.approximate_hessian(function, x0, dx))
            H_inv = np.linalg.inv(self.approximate_hessian(function, x0, dx))
            grad = gradient(function, x0, dx)
            Sk = -1*H_inv @ grad
            x1 = x0 + Sk
            # Backup termination
            i += 1
            if i > max_iterations:
                print("Maximum iterations achieved without convergence")
                break
        return x1
