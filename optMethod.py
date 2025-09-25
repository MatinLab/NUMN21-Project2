import numpy as np
from numpy.linalg import LinAlgError
from optProblem import OptProblem

class OptMethod:
    """Base class for optimization methods."""
    def optimize(self, function, gradient=None, x0=None, **kwargs):
        raise NotImplementedError


class Newton(OptMethod):
    """
    Newton's method with FD Hessian (Task 3) + Exact Line Search (Task 4).
    - Hessian via finite differences (from OptProblem), symmetrized and regularized.
    - Exact line search along p using bracket + golden-section.
    """
    def __init__(
        self,
        tol=1e-8,
        max_iter=100,
        h=None,
        reg=1e-8,
        callback=None,
        # line search params
        use_line_search=True,
        bracket_expand=2.0,
        max_bracket_iters=20,
        golden_tol=1e-6,
        max_golden_iters=60
    ):
        self.tol = tol
        self.max_iter = max_iter
        self.h = h
        self.reg = reg
        self.callback = callback

        self.use_line_search = use_line_search
        self.bracket_expand = float(bracket_expand)
        self.max_bracket_iters = int(max_bracket_iters)
        self.golden_tol = float(golden_tol)
        self.max_golden_iters = int(max_golden_iters)

    def _as_problem(self, function, gradient):
        if isinstance(function, OptProblem):
            return function
        # Wrap bare callables in an OptProblem to unify interface
        return OptProblem(function, gradient)

    # ----- Task 4: Exact line search helpers -----
    @staticmethod
    def _phi(problem: OptProblem, x: np.ndarray, p: np.ndarray, alpha: float) -> float:
        return problem.value(x + alpha * p)

    def _bracket_minimum(self, problem, x, p, f0, alpha0=1.0):
        """
        Expand alpha by factor until phi increases: find [a_lo, a_hi] s.t. phi decreases then increases.
        Returns (a_lo, a_hi). If no increase found within budget, returns (0, alpha_last).
        """
        a_prev = 0.0
        f_prev = f0

        alpha = float(alpha0)
        f_curr = self._phi(problem, x, p, alpha)

        # If first try is not better, shrink towards zero to find a small improving step
        shrink_tries = 0
        while f_curr >= f_prev and alpha > 1e-16 and shrink_tries < 8:
            alpha *= 0.5
            f_curr = self._phi(problem, x, p, alpha)
            shrink_tries += 1

        if f_curr >= f_prev:
            # Could not find a decreasing direction -> just fall back to tiny step
            return (0.0, max(alpha, 1e-8))

        # Now expand until we see an increase to form a bracket
        for _ in range(self.max_bracket_iters):
            a_next = alpha * self.bracket_expand
            f_next = self._phi(problem, x, p, a_next)

            if f_next > f_curr:
                # Bracket found between alpha and a_next (and also from 0 to a_next)
                return (0.0, a_next)

            # keep expanding
            a_prev, f_prev = alpha, f_curr
            alpha, f_curr = a_next, f_next

        # Monotone decrease within budget -> take last interval
        return (0.0, alpha)

    def _golden_section(self, problem, x, p, a_lo, a_hi):
        """
        Golden-section search to (near-)exactly minimize phi on [a_lo, a_hi].
        """
        phi = lambda a: self._phi(problem, x, p, a)

        invphi = (np.sqrt(5.0) - 1.0) / 2.0  # ~0.618
        invphi2 = (3.0 - np.sqrt(5.0)) / 2.0  # ~0.382

        lo, hi = float(a_lo), float(a_hi)
        # Initialize internal points
        a1 = lo + invphi2 * (hi - lo)
        a2 = lo + invphi  * (hi - lo)
        f1 = phi(a1)
        f2 = phi(a2)

        it = 0
        while (hi - lo) > self.golden_tol and it < self.max_golden_iters:
            if f1 > f2:
                lo = a1
                a1 = a2
                f1 = f2
                a2 = lo + invphi * (hi - lo)
                f2 = phi(a2)
            else:
                hi = a2
                a2 = a1
                f2 = f1
                a1 = lo + invphi2 * (hi - lo)
                f1 = phi(a1)
            it += 1

        # Best is the midpoint (or min of endpoints)
        a_star = 0.5 * (lo + hi)
        f_star = phi(a_star)
        return a_star, f_star
    # ----- end Task 4 helpers -----

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
            G = 0.5 * (G + G.T)

            # Regularize if ill-conditioned / singular
            lam = self.reg
            I = np.eye(G.shape[0])
            for _ in range(6):  # up to 6 attempts to make it PD/invertible
                try:
                    p = np.linalg.solve(G + lam * I, -g)
                    break
                except LinAlgError:
                    lam = max(10 * lam, 1e-12)
            else:
                # Fallback: gradient step
                p = -g

            # Ensure descent direction (if numeric quirks)
            if float(np.dot(g, p)) > 0:
                p = -p

            # ----- Task 4: exact line search -----
            if self.use_line_search:
                # Bracket a minimum, then golden-section refine
                a_lo, a_hi = self._bracket_minimum(problem, x, p, f0=fx, alpha0=1.0)
                alpha, _ = self._golden_section(problem, x, p, a_lo, a_hi)
            else:
                alpha = 1.0
            # -------------------------------------

            x = x + alpha * p

        # If we exit the loop without converging:
        fx = problem.value(x)
        g = problem.grad(x, h=self.h)
        return x, fx, {
            "iters": self.max_iter,
            "grad_norm": float(np.linalg.norm(g)),
            "history": hist,
            "status": "max_iter"
        }
