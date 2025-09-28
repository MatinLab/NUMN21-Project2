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
    def approximate_grad(self, f, x, dx=10**-4):
        # Approximate the gradient of a function f about the vector x
        # x must be a numpy array and f must take numpy arrays as arguments
        # dx is the delta to add to each variable
        grad = np.zeros(x.shape, dtype=np.float64)
        f0 = f(x)
        for i in range(len(grad)):
            x1 = x.copy()
            x1[i] += dx
            grad[i] = (f(x1)-f0)/dx
        return grad
    # Numerically approximate the gradient of a 1D function
    def approximate_grad_1D(self, f, x, dx=10**-4):
        # Approximate the gradient of a function f at x
        # x must be a float and f must take a single float as argument
        # dx is the delta to add to x
        return (f(x+dx)-f(x))/dx
    # Function to approximate the Hessian of a function
    def approximate_hessian(self, f, x, dx=10**-4):
        H = np.zeros((len(x), len(x)), dtype=np.float64)
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
    # Binomial search function
    def binary_minimum(self, f, a, b, tol=10**-4, max_iterations=10**5):
        # Search for a minimum of f on the interval [a, b]
        # Perform binary search on the gradient of f
        # 1d function f
        ga = self.approximate_grad_1D(f, a)
        gb = self.approximate_grad_1D(f, b)
        assert not ga*gb>0, "Derivative must change sign in the interval"
        i = 0
        while b-a > tol:
            # If our interval isn't small enough, keep looking
            m = (b+a)/2
            gm = self.approximate_grad_1D(f, m)
            if gm*ga < 0: # sign change between a and m
                # Minimum is between a and m
                b = m
                gb = gm
            elif gm*gb < 0: # sign change between m and b
                # Min between m and b
                a = m
                ga = gm
            else:
                # m is the minimum
                return m
            i += 1
            if i > max_iterations:
                print("Minimum not found to specified tolerance in the maximum number of iterations")
                print(f"At end interval [a, b]=[{a}, {b}]")
                break
        return (b+a)/2
                
        
        
class Newton(OptMethod):
    # Newtons method for finding the root of the gradient
    def optimize(self, function, x0, gradient=None, dx=10**-4, termination_criterion=None, tol=10**-4, alpha=None, max_iterations=10**5):
        """
        Parameters
        ----------
        function :
            Use Newton's method to calculate the minimum of a given function.
        x0 : ndarray
            Starting point.
        gradient : function, optional
            A function which takes a function, a point, and a delta dx. Should return the gradient of the function at the point. The default is None.
        dx : float, optional
            Delta dx for numerically approximating the gradient and/or Hessian. The default is 10**-4.
        termination_criterion : function, optional
            A function which takes the starting point, next point in the method, gradient, inverse Hessian, and tolerance.
            Returns True when termination should occur. The default is None.
        tol : float, optional
            Tolerance within which termination should occur in termination_criterion. The default is 10**-4.
        alpha : float, optional
            Step size in Newton's method. If not specified, exact line search is used. The default is 1.
        max_iterations : int, optional
            Maximum iterations to perform before terminating. The default is 10**5.

        Returns
        -------
        x1 : ndarray
            The point at which a minimum is found.

        """
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
        if alpha is None:
            alpha = self.exact_line_search(function, x0, grad, H_inv, dx, tol)
        x1 = x0 + alpha*Sk
        i = 0
        while not termination_criterion(x0, x1, grad, H_inv, tol):
            # Hasn't converged, make another step
            x0 = x1.copy()
            try:
                H = self.approximate_hessian(function, x0, dx)
                H_inv = np.linalg.inv(H)
            except np.linalg.LinAlgError:
                print(f"LINALGERROR, iteration {i}")
                print(x0)
                print(H)
                exit()
            H_inv = np.linalg.inv(self.approximate_hessian(function, x0, dx))
            grad = gradient(function, x0, dx)
            Sk = -1*H_inv @ grad
            if alpha is None:
                alpha = self.exact_line_search(function, x0, grad, H_inv, dx, tol)
            x1 = x0 + alpha*Sk
            # Backup termination
            i += 1
            if i > max_iterations:
                print("Maximum iterations achieved without convergence")
                break
        return x1
    
    # Exact line search method
    def exact_line_search(self, f, x, grad, H_inv, dx=10**-4, tol=10**-4):
        # Find the minimum of the function f(alpha) = f(x+alpha*(-H_inv@grad))
        # Using binary search
        # Find an interval [0, b] on which the sign of the derivative of f(alpha) changes
        Sk = -1*H_inv @ grad
        f_alpha = lambda a: f(x+a*Sk)
        a = 0
        b = 0.1
        ga = self.approximate_grad_1D(f_alpha, a, dx)
        gb = self.approximate_grad_1D(f_alpha, b, dx)
        while gb*ga > 0: # while no sign change
            a = b
            b += 0.1
            gb = self.approximate_grad_1D(f_alpha, b, dx)
            # Terminate if b has gotten too high
            if b > 5:
                # b too high, use alpha of 1
                #print("Bounds for alpha exceeded, using alpha=1")
                return 1
        # Sign change found at b
        #print(f"sign change found on interval [{a}, {b}]")
        # Find the minimum on the interval a, b
        return self.binary_minimum(f_alpha, a, b, tol)
        
    
