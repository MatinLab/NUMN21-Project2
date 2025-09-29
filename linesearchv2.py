from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Protocol

@dataclass
class LineSearchResult:
    alpha: float
    f_new: float

class LineSearch(Protocol):
    def __call__(self, problem, x, p, f0, g0) -> LineSearchResult: ...

# -------------------- Exact line search (bracket + golden) --------------------

class ExactLineSearch:
    def __init__(self, bracket_expand=2.0, max_bracket_iters=20, golden_tol=1e-6, max_golden_iters=60):
        self.bracket_expand = float(bracket_expand)
        self.max_bracket_iters = int(max_bracket_iters)
        self.golden_tol = float(golden_tol)
        self.max_golden_iters = int(max_golden_iters)

    @staticmethod
    def _phi(problem, x, p, a):
        return problem.value(x + a * p)

    def _bracket(self, problem, x, p, f0, alpha0=1.0):
        a = float(alpha0)
        f_curr = self._phi(problem, x, p, a)

        # Try shrinking if no improvement
        tries = 0
        while f_curr >= f0 and a > 1e-16 and tries < 8:
            a *= 0.5
            f_curr = self._phi(problem, x, p, a)
            tries += 1
        if f_curr >= f0:
            return 0.0, max(a, 1e-8)

        # Expand until it increases
        alpha = a
        f_prev = f0
        for _ in range(self.max_bracket_iters):
            a_next = alpha * self.bracket_expand
            f_next = self._phi(problem, x, p, a_next)
            if f_next > f_curr:
                return 0.0, a_next
            f_prev = f_curr
            alpha, f_curr = a_next, f_next
        return 0.0, alpha

    def _golden(self, problem, x, p, a_lo, a_hi):
        phi = lambda a: self._phi(problem, x, p, a)
        invphi = (np.sqrt(5.0) - 1.0) / 2.0
        invphi2 = (3.0 - np.sqrt(5.0)) / 2.0

        lo, hi = float(a_lo), float(a_hi)
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
        a_star = 0.5 * (lo + hi)
        return a_star, phi(a_star)

    def __call__(self, problem, x, p, f0, g0):
        a_lo, a_hi = self._bracket(problem, x, p, f0, alpha0=1.0)
        a_star, f_star = self._golden(problem, x, p, a_lo, a_hi)
        return LineSearchResult(alpha=float(a_star), f_new=float(f_star))

# -------------------- Strong Wolfe line search (Armijo + curvature) --------------------

class WolfeLineSearch:
    """
    Strong Wolfe conditions:
      1) Armijo (sufficient decrease): phi(a) <= phi(0) + c1 * a * phi'(0)
      2) Curvature: |phi'(a)| <= c2 * |phi'(0)|
    Uses the standard bracket+zoom routine (Nocedal & Wright).
    """
    def __init__(self, c1=1e-4, c2=0.9, alpha0=1.0, expand=2.0, max_it=40):
        assert 0 < c1 < c2 < 1
        self.c1 = float(c1)
        self.c2 = float(c2)
        self.alpha0 = float(alpha0)
        self.expand = float(expand)
        self.max_it = int(max_it)

    @staticmethod
    def _phi(problem, x, p, a):
        return problem.value(x + a * p)

    @staticmethod
    def _dphi(problem, x, p, a):
        g = problem.grad(x + a * p)
        return float(np.dot(g, p))

    def __call__(self, problem, x, p, f0, g0):
        # Scalars for the 1D function and its slope at 0
        phi0 = float(f0)
        dphi0 = float(np.dot(g0, p))  # should be < 0 for descent

        # If not a descent direction, flip (defensive)
        if dphi0 >= 0.0:
            p = -p
            dphi0 = -dphi0

        alpha_prev = 0.0
        phi_prev = phi0
        alpha = self.alpha0

        for i in range(self.max_it):
            phi = self._phi(problem, x, p, alpha)

            # Armijo violation or not improved over previous best in bracket
            if (phi > phi0 + self.c1 * alpha * dphi0) or (i > 0 and phi >= phi_prev):
                return self._zoom(problem, x, p, phi0, dphi0, alpha_prev, alpha)

            dphi = self._dphi(problem, x, p, alpha)

            # Curvature condition satisfied
            if abs(dphi) <= self.c2 * abs(dphi0):
                return LineSearchResult(alpha=float(alpha), f_new=float(phi))

            # If slope is positive, minimum lies between alpha_prev and alpha
            if dphi >= 0:
                return self._zoom(problem, x, p, phi0, dphi0, alpha, alpha_prev)

            # Otherwise, expand the interval and continue
            alpha_prev = alpha
            phi_prev = phi
            alpha *= self.expand

        # Fallback: return last tried alpha
        return LineSearchResult(alpha=float(alpha), f_new=float(self._phi(problem, x, p, alpha)))

    def _zoom(self, problem, x, p, phi0, dphi0, alo, ahi):
        # Ensure ordering
        lo, hi = (alo, ahi) if alo < ahi else (ahi, alo)

        for _ in range(self.max_it):
            amid = 0.5 * (lo + hi)
            phi_mid = self._phi(problem, x, p, amid)
            phi_lo  = self._phi(problem, x, p, lo)

            if (phi_mid > phi0 + self.c1 * amid * dphi0) or (phi_mid >= phi_lo):
                hi = amid
            else:
                dphi_mid = self._dphi(problem, x, p, amid)
                if abs(dphi_mid) <= self.c2 * abs(dphi0):
                    return LineSearchResult(alpha=float(amid), f_new=float(phi_mid))
                if dphi_mid * (hi - lo) >= 0:
                    hi = lo
                lo = amid

            if abs(hi - lo) < 1e-12:
                return LineSearchResult(alpha=float(amid), f_new=float(phi_mid))

        # As a last resort
        amid = 0.5 * (lo + hi)
        return LineSearchResult(alpha=float(amid), f_new=float(self._phi(problem, x, p, amid)))
