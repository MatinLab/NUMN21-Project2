import numpy as np
from linesearch import InexactLineSearchMethod

class OptMethod():
    # Generic class for Quasi Newton methods
    # Constructor
    def __init__(self):
        self.H = None # Store current Hessian approximation
        self.H_inv = None # Store current Hessian inverse approximation
    # Optimize method
    def optimize(self, function, gradient=None):
        # To be overridden in the child class
        pass
    # Numerically approximate the gradient of a function
    def approximate_grad(self, f, x, dx=10**-4):
        # Approximate the gradient of a function f about the vector x
        # x must be a numpy array and f must take numpy arrays as arguments
        # dx is the delta to add to each variable
        grad = np.zeros(x.shape, dtype=np.double)
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
    # Function to instantiate the inverse of the Hessian using a numerical approximation
    # This can be overriden as desired separately from approximate_hessian below
    # though they start the same
    def instantiate_hessian_inv(self, f, x, gradient=None, dx=10**-4, x0=None):
        # Accept optional parameter x0 for compatability with Quasi Newton approximations
        if gradient is None:
            gradient = self.approximate_grad
        H = np.zeros((len(x), len(x)), dtype=np.double)
        grad0 = gradient(f, x, dx)
        for i in range(len(x)):
            # For each xi, we get the derivative of the gradient with
            # respect to the xi
            # This is the ith row in the Hessian
            x1 = x.copy()
            x1[i] += dx
            H[i] = (gradient(f, x1, dx)-grad0)/dx
        H_inv = np.linalg.inv(H)
        # Store Hessians
        self.H = H
        self.H_inv = H_inv
        return H_inv
    # Function to numerically approximate the inverse of the Hessian of a function
    def approximate_hessian_inv(self, f, x, gradient=None, dx=10**-4, x0=None):
        # Accept optional parameter x0 for compatability with Quasi Newton approximations
        if gradient is None:
            gradient = self.approximate_grad
        H = np.zeros((len(x), len(x)), dtype=np.double)
        grad0 = gradient(f, x, dx)
        for i in range(len(x)):
            # For each xi, we get the derivative of the gradient with
            # respect to the xi
            # This is the ith row in the Hessian
            x1 = x.copy()
            x1[i] += dx
            H[i] = (gradient(f, x1, dx)-grad0)/dx
        H_inv = np.linalg.inv(H)
        # Store Hessians
        self.H = H
        self.H_inv = H_inv
        return H_inv
    
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
    
    # Line search methods
    def line_search(self, f, x, gradient, H_inv, dx, tol):
        # Follow this format when defining a line search method
        # Then specify that you want to use it as a parameter in optimize
        # Don't need self if the function is defined outside a class
        pass
    
    # Exact line search method
    def exact_line_search(self, f, x, gradient, H_inv, dx=10**-4, tol=10**-4):
        # Find the minimum of the function f(alpha) = f(x+alpha*(-H_inv@grad))
        # Using binary search
        # Find an interval [0, b] on which the sign of the derivative of f(alpha) changes
        grad = gradient(f, x, dx)
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
            if b > 3:
                # b too high, use alpha of 1
                #print("Bounds for alpha exceeded, using alpha=1")
                return 1
        # Sign change found at b
        #print(f"sign change found on interval [{a}, {b}]")
        # Find the minimum on the interval a, b
        return self.binary_minimum(f_alpha, a, b, tol)
    
    # Inexact line search wrapper
    def create_inexact_linesearch(self, f, alpha_init, dx=10**-4, tol=10**-4, f_bar=-np.inf, rho=1e-2, sigma=0.1, tau=0.9, bracketing_max_iterations=50,
                              tau2=0.1, tau3=0.5, sectioning_max_iterations=10):
    # Create a lambda that will match the format required by the code in the Newton methods
        def line_search(f, x, gradient, H_inv, dx=10**-4, tol=10**-4):
            # Compute the search direction at the current point
            grad = gradient(f, x, dx)
            direction = -H_inv @ grad
            
            #print(f"Line search called at x={x}")
            #print(f"Direction: {direction}")
            #print(f"Gradient norm: {np.linalg.norm(grad)}")
            
            # Call inexact line search with the current direction
            alpha = InexactLineSearchMethod(
                f, 
                lambda pt: gradient(f, pt, dx), 
                alpha_init, 
                direction, 
                x, 
                f_bar, 
                rho, 
                sigma, 
                tau, 
                bracketing_max_iterations, 
                tau2, 
                tau3, 
                sectioning_max_iterations
            )
            return alpha
    
        return line_search
             
                
        
# Newton's method      
class Newton(OptMethod):
    # Newtons method for finding the root of the gradient
    def optimize(self, function, x0, gradient=None, dx=10**-4, termination_criterion=None, tol=10**-4, alpha=None, line_search=None, max_iterations=10**5):
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
        line_search: function, optional
            Function to return the alpha step size. If not specified, exact line search is used. Any line search function
            should take the function, the point, the gradient, the inverse Hessian, a step size to use in numerical
            approximations, and the tolerance with which to find the step size. The default is None.
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
        # Default to exact line search
        if line_search is None:
            line_search = self.exact_line_search
        # Run Newton's method
        H_inv = self.instantiate_hessian_inv(function, x0, gradient, dx)
        #H_inv = np.linalg.inv(self.H)
        #grad = gradient(function, x0, dx)
        #Sk = -1*H_inv @ grad
        #if alpha is None:
            #alpha = line_search(function, x0, grad, H_inv, dx, tol)
        #x1 = x0 + alpha*Sk
        i = 0
        while True:
            # Hasn't converged, make another step
            grad = gradient(function, x0, dx)
            Sk = -1*H_inv @ grad
            if alpha is None:
                alpha_k = line_search(function, x0, gradient, H_inv, dx, tol)
                #print(f"Alpha for step {i}: {alpha_k}")
                #if i>20:
                    #break
            else:
                alpha_k = alpha
            x1 = x0 + alpha_k*Sk
            # Check termination condition
            if termination_criterion(x0, x1, grad, H_inv, tol):
                break
            # else we continue
            # Update Hessian for the next step
            H_inv = self.approximate_hessian_inv(function, x1, gradient, dx, x0)
            # Set new x0
            x0 = x1.copy()
            # Backup termination
            i += 1
            if i > max_iterations:
                print("Maximum iterations achieved without convergence")
                break
        return x1
        
# Quasi Newton methods

# Good Broyden method
class GoodBroyden(Newton):
    # Perform Newton's method style steps, but approximate the Hessian using a "good" Broyden rank 1 update and
    # the Sherman-Morisson computation of the new matrix
    # Broyden update to the Hessian
    def broyden_update(self, function, x0, x1, gradient, dx):
        # Define delta and lambda values as
        # deltak = xk+1-xk and lambdak=gradk+1-gradk
        delta_k = x1-x0
        lambda_k = gradient(function, x1, dx)-gradient(function, x0, dx)
        # Reshaping vectors
        delta_k = delta_k.reshape((-1, 1))
        lambda_k = lambda_k.reshape((-1, 1))
        H_inv = self.H_inv
        # Then use Sherman-Morisson formula
        summed_value = ((delta_k-(H_inv @ lambda_k))/(delta_k.T @ H_inv @ lambda_k)) @ (delta_k.T @ H_inv)
        Hkplus1_inv = H_inv + summed_value
        Hkplus1 = np.linalg.inv(Hkplus1_inv)
        return Hkplus1, Hkplus1_inv
    # Override the approximate_hessian_inv function
    def approximate_hessian_inv(self, f, x, gradient=None, dx=10**-4, x0=None):
        assert x0 is not None, "Previous point in the method required for Broyden step"
        H, H_inv = self.broyden_update(f, x0, x, gradient, dx)
        self.H = H
        self.H_inv = H_inv
        return H_inv

# Bad Broyden method
class BadBroyden(Newton):
    # Perform Newton's method style steps, but approximate the Hessian using a "bad" Broyden rank 1 update
    # Broyden update to the Hessian
    def broyden_update(self, function, x0, x1, gradient, dx):
        # Define delta and lambda values as
        # deltak = xk+1-xk and lambdak=gradk+1-gradk
        delta_k = x1-x0
        lambda_k = gradient(function, x1, dx)-gradient(function, x0, dx)
        # Reshaping vectors
        delta_k = delta_k.reshape((-1, 1))
        lambda_k = lambda_k.reshape((-1, 1))
        H_inv = self.H_inv
        # Then use bad broyden update
        Hkplus1_inv = H_inv + ((delta_k - (H_inv @ lambda_k))/(lambda_k.T @ lambda_k)) @ lambda_k.T
        Hkplus1 = np.linalg.inv(Hkplus1_inv)
        return Hkplus1, Hkplus1_inv
    # Override the approximate_hessian_inv function
    def approximate_hessian_inv(self, f, x, gradient=None, dx=10**-4, x0=None):
        assert x0 is not None, "Previous point in the method required for Broyden step"
        H, H_inv = self.broyden_update(f, x0, x, gradient, dx)
        self.H = H
        self.H_inv = H_inv
        return H_inv


# Symmetric Broyden method
class SymmetricBroyden(Newton):
    # Perform Newton's method style steps, but approximate the Hessian using a symmetric Broyden rank 1 update
    # Broyden update to the Hessian
    def broyden_update(self, function, x0, x1, gradient, dx):
        # Define delta and lambda values as
        # deltak = xk+1-xk and lambdak=gradk+1-gradk
        delta_k = x1-x0
        lambda_k = gradient(function, x1, dx)-gradient(function, x0, dx)
        # Reshaping vectors
        delta_k = delta_k.reshape((-1, 1))
        lambda_k = lambda_k.reshape((-1, 1))
        H_inv = self.H_inv
        # Define u and a
        u_k = delta_k - (H_inv @ lambda_k)
        a_k = 1 / (u_k.T @ lambda_k)
        # Make a symmetric update
        Hkplus1_inv = H_inv + (a_k * (u_k @ u_k.T))
        Hkplus1 = np.linalg.inv(Hkplus1_inv)
        return Hkplus1, Hkplus1_inv
    # Override the approximate_hessian_inv function
    def approximate_hessian_inv(self, f, x, gradient=None, dx=10**-4, x0=None):
        assert x0 is not None, "Previous point in the method required for Broyden step"
        H, H_inv = self.broyden_update(f, x0, x, gradient, dx)
        self.H = H
        self.H_inv = H_inv
        return H_inv

# DFP method
class DFP(Newton):
    # Perform Newton's method style steps, but approximate the Hessian using a DFP rank 2 update
    # DFP update to the Hessian
    def dfp_update(self, function, x0, x1, gradient, dx):
        # Define delta and lambda values as
        # deltak = xk+1-xk and lambdak=gradk+1-gradk
        delta_k = x1-x0
        lambda_k = gradient(function, x1, dx)-gradient(function, x0, dx)
        # Reshaping vectors
        delta_k = delta_k.reshape((-1, 1))
        lambda_k = lambda_k.reshape((-1, 1))
        H_inv = self.H_inv
        # Make a rank 2 update
        term1 = (delta_k @ delta_k.T)/(delta_k.T @ lambda_k)
        term2 = (H_inv @ lambda_k @ lambda_k.T @ H_inv)/(lambda_k.T @ H_inv @ lambda_k)
        Hkplus1_inv = H_inv + term1 - term2
        Hkplus1 = np.linalg.inv(Hkplus1_inv)
        return Hkplus1, Hkplus1_inv
    # Override the approximate_hessian_inv function
    def approximate_hessian_inv(self, f, x, gradient=None, dx=10**-4, x0=None):
        assert x0 is not None, "Previous point in the method required for DFP step"
        H, H_inv = self.dfp_update(f, x0, x, gradient, dx)
        self.H = H
        self.H_inv = H_inv
        return H_inv

# BFGS method
class BFGS(Newton):
    # Perform Newton's method style steps, but approximate the Hessian using a BFGS rank 2 update
    # BFGS update to the Hessian
    def bfgs_update(self, function, x0, x1, gradient, dx):
        # Define delta and lambda values as
        # deltak = xk+1-xk and lambdak=gradk+1-gradk
        delta_k = x1-x0
        lambda_k = gradient(function, x1, dx)-gradient(function, x0, dx)
        # Reshaping vectors
        delta_k = delta_k.reshape((-1, 1))
        lambda_k = lambda_k.reshape((-1, 1))
        H_inv = self.H_inv
        # Make a rank 2 update
        term1_factor = 1 + (lambda_k.T @ H_inv @ lambda_k)/(delta_k.T @ lambda_k)
        term1 = (delta_k @ delta_k.T)/(delta_k.T @ lambda_k)
        term2 = ((delta_k @ lambda_k.T @ H_inv) + (H_inv @ lambda_k @ delta_k.T))/(delta_k.T @ lambda_k)
        Hkplus1_inv = H_inv + (term1_factor * term1) - term2
        Hkplus1 = np.linalg.inv(Hkplus1_inv)
        return Hkplus1, Hkplus1_inv
    # Override the approximate_hessian_inv function
    def approximate_hessian_inv(self, f, x, gradient=None, dx=10**-4, x0=None):
        assert x0 is not None, "Previous point in the method required for BFGS step"
        H, H_inv = self.bfgs_update(f, x0, x, gradient, dx)
        self.H = H
        self.H_inv = H_inv
        return H_inv