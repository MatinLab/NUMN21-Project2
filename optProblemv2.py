import numpy as np

class OptProblem:
    """
    Optimization problem wrapper.
    - f(x): objective (R^n -> R)
    - grad(x): optional gradient; if None, use central finite differences
    Provides finite-difference Hessian with symmetrization.
    Also tracks evaluation counters: n_f, n_g, n_H.
    """
    def __init__(self, f, grad=None):
        self.f = f
        self.grad_fun = grad
        self.n_f = 0
        self.n_g = 0
        self.n_H = 0

    def value(self, x: np.ndarray) -> float:
        x = np.asarray(x, dtype=float)
        self.n_f += 1
        return float(self.f(x))

    def grad(self, x: np.ndarray, h=None) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        if self.grad_fun is not None:
            self.n_g += 1
            return np.asarray(self.grad_fun(x), dtype=float)

        # Central-difference gradient
        self.n_g += 1  # count the "gradient evaluation" at high level
        n = x.size
        g = np.zeros(n, dtype=float)
        if h is None:
            eps = np.finfo(float).eps
            h = np.sqrt(eps)

        # Note: counts f-evals inside value(); we keep n_g independent
        for i in range(n):
            ei = np.zeros(n); ei[i] = 1.0
            g[i] = (self.value(x + h*ei) - self.value(x - h*ei)) / (2*h)
        return g

    def hess_fd(self, x: np.ndarray, h=None) -> np.ndarray:
        """
        Finite-difference Hessian using only f.
        Diagonals:   [f(x+he_i) - 2f(x) + f(x-he_i)] / h^2
        Off-diagon.: [f(x+he_i+he_j) - f(x+he_i-he_j)
                      - f(x-he_i+he_j) + f(x-he_i-he_j)] / (4 h^2)
        Then symmetrize: 0.5 * (G + G.T)
        """
        x = np.asarray(x, dtype=float)
        n = x.size
        G = np.zeros((n, n), dtype=float)

        if h is None:
            eps = np.finfo(float).eps
            h = (np.sqrt(eps) * (1.0 + np.abs(x))).mean()

        self.n_H += 1
        f0 = self.value(x)

        # Diagonal terms
        for i in range(n):
            ei = np.zeros(n); ei[i] = 1.0
            fpp = self.value(x + h*ei)
            fmm = self.value(x - h*ei)
            G[i, i] = (fpp - 2.0*f0 + fmm) / (h*h)

        # Off-diagonal terms (i < j)
        for i in range(n):
            ei = np.zeros(n); ei[i] = 1.0
            for j in range(i+1, n):
                ej = np.zeros(n); ej[j] = 1.0
                fpp = self.value(x + h*ei + h*ej)
                fpm = self.value(x + h*ei - h*ej)
                fmp = self.value(x - h*ei + h*ej)
                fmm = self.value(x - h*ei - h*ej)
                val = (fpp - fpm - fmp + fmm) / (4*h*h)
                G[i, j] = val
                G[j, i] = val

        return 0.5 * (G + G.T)
