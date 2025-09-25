import numpy as np

class OptProblem:
    """
    Optimization problem wrapper.
    - f(x): objective (R^n -> R)
    - grad(x): optional gradient; if None, use finite differences
    """
    def __init__(self, f, grad=None):
        self.f = f
        self.grad_fun = grad

    def value(self, x: np.ndarray) -> float:
        return float(self.f(np.asarray(x, dtype=float)))

    def grad(self, x: np.ndarray, h) -> np.ndarray:
        if self.grad_fun is not None:
            return np.asarray(self.grad_fun(np.asarray(x, dtype=float)), dtype=float)

        # Central-difference gradient
        x = np.asarray(x, dtype=float)
        n = x.size
        g = np.zeros(n, dtype=float)
        if h is None:
            eps = np.finfo(float).eps
            h = np.sqrt(eps)

        for i in range(n):
            ei = np.zeros(n); ei[i] = 1.0
            g[i] = (self.f(x + h*ei) - self.f(x - h*ei)) / (2*h)
        return g

    def hess_fd(self, x: np.ndarray, h) -> np.ndarray:
        """
        Finite-difference Hessian using only f (Task 3).
        - Diagonals:   f(x+he_i) - 2f(x) + f(x-he_i)        / h^2
        - Off-diagon.: f(x+he_i+he_j) - f(x+he_i-he_j)
                        - f(x-he_i+he_j) + f(x-he_i-he_j)   / (4 h^2)
        Then symmetrize: 0.5 * (G + G.T)
        """
        x = np.asarray(x, dtype=float)
        n = x.size
        G = np.zeros((n, n), dtype=float)

        if h is None:
            # Step size scaled to x
            eps = np.finfo(float).eps
            h = (np.sqrt(eps) * (1.0 + np.abs(x))).mean()

        f0 = self.f(x)

        # Diagonal terms
        for i in range(n):
            ei = np.zeros(n); ei[i] = 1.0
            fpp = self.f(x + h*ei)
            fmm = self.f(x - h*ei)
            G[i, i] = (fpp - 2.0*f0 + fmm) / (h*h)

        # Off-diagonal terms (i < j)
        for i in range(n):
            ei = np.zeros(n); ei[i] = 1.0
            for j in range(i+1, n):
                ej = np.zeros(n); ej[j] = 1.0
                fpp = self.f(x + h*ei + h*ej)
                fpm = self.f(x + h*ei - h*ej)
                fmp = self.f(x - h*ei + h*ej)
                fmm = self.f(x - h*ei - h*ej)
                val = (fpp - fpm - fmp + fmm) / (4*h*h)
                G[i, j] = val
                G[j, i] = val  # fill symmetrically

        # Symmetrize explicitly anyway (as required by Task 3)
        G = 0.5 * (G + G.T)
        return G
