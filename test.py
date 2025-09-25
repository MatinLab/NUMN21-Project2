import unittest

import numpy as np
from optProblem import OptProblem
from optMethod import Newton
import matplotlib.pyplot as plt

def rosen(x):
    x = np.asarray(x, dtype=float)
    return 100.0*(x[1]-x[0]**2)**2 + (1.0 - x[0])**2

# Optional analytic gradient (not required; your code can FD this)
def rosen_grad(x):
    x = np.asarray(x, dtype=float)
    dfdx = -400.0 * x[0] * (x[1] - x[0]**2) - 2.0*(1.0 - x[0])
    dfdy =  200.0 * (x[1] - x[0]**2)
    return np.array([dfdx, dfdy], dtype=float)

def run_newton_and_plot(info,
                        x_range=(-1.5, 2.0),
                        y_range=(-0.5, 3.0),
                        n_grid=400,
                        levels=None,
                        show=False):
    """
    Plot contour of Rosenbrock + path of iterates from Newton info.
    """
    # 1) Extract path from info
    path = np.array([rec["x"] for rec in info["history"]] + [info["history"][-1]["x"]])

    # 2) Prepare contour grid
    xg = np.linspace(x_range[0], x_range[1], n_grid)
    yg = np.linspace(y_range[0], y_range[1], n_grid)
    X, Y = np.meshgrid(xg, yg)
    Z = 100.0 * (Y - X**2)**2 + (1.0 - X)**2

    if levels is None:
        levels = np.geomspace(1e-2, 1e4, 20)

    # 3) Plot
    plt.figure(figsize=(7, 6))
    cs = plt.contour(X, Y, Z, levels=levels)
    plt.clabel(cs, inline=True, fontsize=8, fmt="%.3g")

    plt.plot(path[:, 0], path[:, 1], marker='o', linewidth=1.5)

    plt.title("Rosenbrock: Newton with exact line search (Task 5)")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.xlim(x_range)
    plt.ylim(y_range)
    plt.grid(True)

    if show:
        plt.show()
    plt.close()


if __name__ == "__main__":

    print("""
========================================================
Testing Classic Newton's Optimization Method (Task 1-4)
========================================================""")
    prob = OptProblem(rosen)
    solver = Newton(tol=1e-8, max_iter=5, use_line_search=True)

    x0 = np.array([-1.2, 1.0])
    xmin, fmin, info = solver.optimize(prob, x0=x0)
    print("x* =", xmin)
    print("f* =", fmin)
    print("status:", info["status"], "iters:", info["iters"], "||g||:", info["grad_norm"])
    print("""
=========================================================================
Testing Performance of Classical Newton's Method with Rosenbrock (Task 5) (needs update)
=========================================================================""")
    run_newton_and_plot(info=info, show=True)
