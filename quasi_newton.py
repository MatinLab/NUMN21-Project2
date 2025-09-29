import numpy as np
from optProblem import OptProblem
from linesearch import LineSearch, ExactLineSearch, LineSearchResult
from optMethod import OptMethod

class QuasiNewton(OptMethod):
    """
    Generic Quasi-Newton framework (inverse-Hessian form by default).
    - Maintains an approximation to inverse Hessian H_k (unless stated otherwise).
    - Direction: p_k = -H_k g_k   (fallback to -g if not descent).
    - Line search: any LineSearch strategy (Exact or Wolfe).
    """
    def __init__(
        self,
        tol: float = 1e-8,
        max_iter: int = 200,
        callback=None,
        line_search: LineSearch | None = None
    ):
        self.tol = tol
        self.max_iter = max_iter
        self.callback = callback
        self.line_search = line_search or ExactLineSearch()

    def _as_problem(self, function, gradient):
        if isinstance(function, OptProblem):
            return function
        return OptProblem(function, gradient)

    def init_inverse_hessian(self, n: int) -> np.ndarray:
        """Default: identity as initial inverse Hessian."""
        return np.eye(n)

    # ---- Virtual update ----
    def update(self, Hk: np.ndarray, s: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Override in subclasses to update inverse Hessian Hk -> H_{k+1}.
        Args:
            Hk: current inverse Hessian approximation
            s: step = x_{k+1} - x_k
            y: grad_{k+1} - grad_k
        """
        raise NotImplementedError

    def optimize(self, function, gradient=None, x0=None, **kwargs):
        if x0 is None:
            raise ValueError("Provide initial guess x0.")

        problem = self._as_problem(function, gradient)
        max_iter = int(kwargs.get("max_iter", self.max_iter))

        x = np.asarray(x0, dtype=float)
        n = x.size
        Hk = self.init_inverse_hessian(n)

        # initial evaluations
        g = problem.grad(x)  # uses default h if analytic grad not provided
        fx = problem.value(x)

        history = []
        for k in range(max_iter):
            gnorm = float(np.linalg.norm(g))
            history.append({
                "iter": k, "x": x.copy(), "f": float(fx), "gnorm": gnorm,
                "n_f": problem.n_f if hasattr(problem, "n_f") else None,
                "n_g": problem.n_g if hasattr(problem, "n_g") else None,
                "n_H": problem.n_H if hasattr(problem, "n_H") else None,
            })
            if self.callback:
                self.callback(k, x, fx, g)

            if gnorm <= self.tol:
                return x, fx, {"iters": k, "grad_norm": gnorm, "history": history, "status": "converged"}

            # direction
            p = -Hk.dot(g)
            if float(np.dot(g, p)) >= 0:
                p = -g  # defensive fallback

            # line search
            ls_res: LineSearchResult = self.line_search(problem, x, p, fx, g)
            alpha = float(ls_res.alpha)
            history[-1]["alpha"] = alpha

            # step
            s = alpha * p
            x_new = x + s

            g_new = problem.grad(x_new)
            y = g_new - g
            fx_new = problem.value(x_new)

            # update inverse Hessian
            Hk = self.update(Hk, s, y)

            x, g, fx = x_new, g_new, fx_new

        return x, fx, {"iters": max_iter, "grad_norm": float(np.linalg.norm(g)), "history": history, "status": "max_iter"}


# =========================
# Rank-2: DFP and BFGS
# =========================

class DFP(QuasiNewton):
    """
    DFP rank-2 update (inverse Hessian form):
        H_{k+1} = Hk + (s s^T)/(s^T y) - (Hk y y^T Hk)/(y^T Hk y)
    """
    def update(self, Hk, s, y):
        sty = float(np.dot(s, y))
        if sty <= 1e-12:
            return Hk  # skip if curvature condition violated
        Hy = Hk.dot(y)
        yHy = float(np.dot(y, Hy))
        # guard (rare): if yHy ~ 0, skip second term
        term1 = np.outer(s, s) / sty
        term2 = (Hy[:, None] @ Hy[None, :]) / max(yHy, 1e-16)
        return Hk + term1 - term2


class BFGS(QuasiNewton):
    """
    BFGS rank-2 update (inverse Hessian form):
        rho = 1/(y^T s)
        H_{k+1} = (I - rho s y^T) Hk (I - rho y s^T) + rho s s^T
    """
    def update(self, Hk, s, y):
        sty = float(np.dot(s, y))
        if sty <= 1e-12:
            return Hk
        rho = 1.0 / sty
        I = np.eye(len(s))
        V = I - rho * np.outer(s, y)
        return V @ Hk @ V.T + rho * np.outer(s, s)


# =========================
# Rank-1: Broyden’s updates
# =========================

class BadBroydenH(QuasiNewton):
    """
    Simple Broyden rank-1 update of the inverse Hessian ("bad Broyden"):
        H_{k+1} = Hk + ((s - Hk y) y^T) / (y^T y)
    """
    def update(self, Hk, s, y):
        yTy = float(np.dot(y, y))
        if yTy <= 1e-16:
            return Hk
        u = s - Hk.dot(y)     # (n,)
        return Hk + np.outer(u, y) / yTy


class GoodBroydenG(QuasiNewton):
    """
    Simple Broyden rank-1 update of the Hessian approximation G ("good Broyden")
    with Sherman–Morrison to update its inverse H = G^{-1} cheaply.

    Good Broyden on G:
        G_{k+1} = Gk + ((y - Gk s) s^T) / (s^T s)

    Sherman–Morrison for (G + u v^T)^{-1}:
        (A + u v^T)^{-1} = A^{-1} - (A^{-1} u v^T A^{-1}) / (1 + v^T A^{-1} u)

    We maintain Hk ≈ Gk^{-1} only; to form u = y - Gk s, we compute t = Gk s by solving
        Hk t = s   (since Hk = Gk^{-1})
    """
    def init_inverse_hessian(self, n: int) -> np.ndarray:
        # Start from G0 = I => H0 = I
        return np.eye(n)

    def update(self, Hk, s, y):
        sTs = float(np.dot(s, s))
        if sTs <= 1e-16:
            return Hk

        # t = Gk s by solving Hk t = s
        # (we avoid forming Gk explicitly)
        try:
            t = np.linalg.solve(Hk, s)  # t = Gk s
        except np.linalg.LinAlgError:
            # If Hk is singular/ill-conditioned, fall back to identity
            t = s.copy()

        u = y - t                             # u = y - Gk s
        v = s / sTs                           # v = s / (s^T s)

        # Sherman–Morrison update of Hk for A = Gk, u, v
        Hu = Hk @ u
        vTHu = float(np.dot(v, Hu))
        denom = 1.0 + vTHu
        if abs(denom) <= 1e-16:
            return Hk  # skip update if denominator too small

        correction = (Hu[:, None] @ (v[None, :] @ Hk)) / denom
        return Hk - correction


class SymmetricBroyden(QuasiNewton):
    """
    Symmetric Broyden (inverse form) as a convex combination of DFP and BFGS:
        H_SB(φ) = (1 - φ) * H_DFP + φ * H_BFGS, with φ in [0, 1]
    Default φ = 0.5 (midpoint).
    """
    def __init__(self, phi: float = 0.5, **kwargs):
        super().__init__(**kwargs)
        self.phi = float(np.clip(phi, 0.0, 1.0))
        self._dfp = DFP(tol=self.tol, max_iter=self.max_iter,
                        callback=self.callback, line_search=self.line_search)
        self._bfgs = BFGS(tol=self.tol, max_iter=self.max_iter,
                          callback=self.callback, line_search=self.line_search)

    def update(self, Hk, s, y):
        # compute both updates from same Hk,s,y and blend
        H_dfp = DFP.update(self, Hk, s, y)
        H_bfgs = BFGS.update(self, Hk, s, y)
        return (1.0 - self.phi) * H_dfp + self.phi * H_bfgs
