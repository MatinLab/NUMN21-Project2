import numpy as np
from numpy.linalg import LinAlgError
from optProblemv2 import OptProblem
from linesearchv2 import LineSearch, ExactLineSearch, LineSearchResult

class OptMethod:
    """Base class for optimization methods."""
    def optimize(self, function, gradient=None, x0=None, **kwargs):
        raise NotImplementedError

class Newton(OptMethod):
    """
    Newton with FD Hessian + symmetrization and pluggable Line Search.
    - line_search: any object implementing LineSearch (Exact or Wolfe).
    - Records chosen alpha in history.
    """
    def __init__(
        self,
        tol=1e-8,
        max_iter=100,
        h=None,
        reg=1e-8,
        callback=None,
        line_search=LineSearch
    ):
        self.tol = tol
        self.max_iter = max_iter
        self.h = h
        self.reg = reg
        self.callback = callback
        self.line_search = line_search or ExactLineSearch()

    def _as_problem(self, function, gradient):
        if isinstance(function, OptProblem):
            return function
        return OptProblem(function, gradient)

    def optimize(self, function, gradient=None, x0=None, **kwargs):
        if x0 is None:
            raise ValueError("Provide an initial guess x0 for Newton's method.")
        problem = self._as_problem(function, gradient)

        # Allow per-call override of max_iter (optional convenience)
        max_iter = int(kwargs.get("max_iter", self.max_iter))

        x = np.asarray(x0, dtype=float)
        hist = []

        for k in range(max_iter):
            fx = problem.value(x)
            g = problem.grad(x, h=self.h)
            gnorm = np.linalg.norm(g, ord=2)

            rec = {
                "iter": k,
                "x": x.copy(),
                "f": float(fx),
                "gnorm": float(gnorm),
                "n_f": problem.n_f,
                "n_g": problem.n_g,
                "n_H": problem.n_H,
            }
            hist.append(rec)

            if self.callback:
                self.callback(k, x, fx, g)

            if gnorm <= self.tol:
                return x, fx, {"iters": k, "grad_norm": gnorm, "history": hist, "status": "converged"}

            # Finite-difference Hessian + symmetrize
            G = problem.hess_fd(x, h=self.h)
            G = 0.5 * (G + G.T)

            # Regularize if ill-conditioned / singular
            lam = self.reg
            I = np.eye(G.shape[0])
            for _ in range(6):
                try:
                    p = np.linalg.solve(G + lam * I, -g)
                    break
                except LinAlgError:
                    lam = max(10 * lam, 1e-12)
            else:
                p = -g  # fallback

            # Ensure descent (defensive)
            if float(np.dot(g, p)) > 0:
                p = -p

            # Line search (strategy object)
            ls_res: LineSearchResult = self.line_search(problem, x, p, fx, g)
            alpha = float(ls_res.alpha)
            hist[-1]["alpha"] = alpha

            # Update
            x = x + alpha * p

        fx = problem.value(x)
        g = problem.grad(x, h=self.h)
        return x, fx, {
            "iters": max_iter,
            "grad_norm": float(np.linalg.norm(g)),
            "history": hist,
            "status": "max_iter"
        }
