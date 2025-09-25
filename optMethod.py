import numpy as np
from numpy.linalg import LinAlgError

# Optional import if you want to accept OptProblem directly
try:
    from optProblem import OptProblem
except Exception:
    OptProblem = None

class OptMethod:
    """Base class for optimization methods."""
    def optimize(self, function, gradient=None, x0=None, **kwargs):
        raise NotImplementedError


class Newton(OptMethod):
    """
    Newton's method with FD Hessian (Task 3).
    - Uses step size 1.0 (exact line search will be Task 4)
    - Regularizes Hessian if near-singular
    - Stops on ||grad|| <= tol or max_iter
    Returns: x, f(x), info dict (history, iters, grad_norm)
    """
    def __init__(self, tol=1e-8, max_iter=100, h=None, reg=1e-8, callback=None):
        self.tol = tol
        self.max_iter = max_iter
        self.h = h
        self.reg = reg
        self.callback = callback

    def _as_problem(self, function, gradient):
        if isinstance(function, OptProblem):
            return function
        # If user passed bare callables, wrap them in OptProblem
        return OptProblem(function, gradient)

    def optimize(self, function, gradient=None, x0=None, **kwargs):
        if x0 is None:
            raise ValueError("Provide an initial guess x0 for Newton's method.")

        problem = self._as_problem(function, gradient)
        x = np.asarray(x0, dtype=float)
        hist = []

        for k in range(self.max_iter):
            fx = problem.value(x)
            g = problem.grad(x, h=self.h)
            gnorm = np.linalg.norm(g, ord=2)

            hist.append({"iter": k, "x": x.copy(), "f": fx, "gnorm": gnorm})

            if self.callback:
                self.callback(k, x, fx, g)

            if gnorm <= self.tol:
                return x, fx, {"iters": k, "grad_norm": gnorm, "history": hist, "status": "converged"}

            # Finite-difference Hessian + symmetrize (Task 3)
            G = problem.hess_fd(x, h=self.h)

            # Regularize if ill-conditioned / singular
            lam = self.reg
            I = np.eye(G.shape[0])
            for _ in range(6):  # up to 6 attempts to make it PD/invertible
                try:
                    # Solve G p = -g
                    p = np.linalg.solve(G + lam*I, -g)
                    break
                except LinAlgError:
                    lam = max(10*lam, 1e-12)
            else:
                # As a fallback, use gradient descent step
                p = -g

            # Task 3: pure Newton step (alpha = 1). Line search comes in Task 4.
            x = x + p

        # If we exit the loop without converging:
        fx = problem.value(x)
        g = problem.grad(x, h=self.h)
        return x, fx, {"iters": self.max_iter, "grad_norm": float(np.linalg.norm(g)), "history": hist, "status": "max_iter"}

