import numpy as np
import matplotlib
# Comment the next line if you want interactive windows by default
# matplotlib.use("Agg")
import matplotlib.pyplot as plt

from optProblemv2 import OptProblem
from optMethodv2 import Newton
from linesearchv2 import WolfeLineSearch, ExactLineSearch

def rosen(x):
    x = np.asarray(x, dtype=float)
    return 100.0*(x[1]-x[0]**2)**2 + (1.0 - x[0])**2

def rosen_grad(x):
    x = np.asarray(x, dtype=float)
    dfdx = -400.0 * x[0] * (x[1] - x[0]**2) - 2.0*(1.0 - x[0])
    dfdy =  200.0 * (x[1] - x[0]**2)
    return np.array([dfdx, dfdy], dtype=float)

def run_newton_and_plot(
    info=None,
    x0=np.array([-1.2, 1.0]),
    solver=None,
    prob=None,
    x_range=(-1.5, 2.0),
    y_range=(-0.5, 3.0),
    n_grid=400,
    levels=None,
    savepath="rosen_task5.png",
    show=False
):
    # Solve if info not given
    if info is None:
        if solver is None or prob is None:
            raise ValueError("Need either info=... OR (solver and prob).")
        x_star, f_star, info = solver.optimize(prob, x0=np.asarray(x0, dtype=float))
        print(f"Solver run finished: f*={f_star:.3e}, iters={info['iters']}")

    path = np.array([rec["x"] for rec in info["history"]])

    xg = np.linspace(x_range[0], x_range[1], n_grid)
    yg = np.linspace(y_range[0], y_range[1], n_grid)
    X, Y = np.meshgrid(xg, yg)
    Z = 100.0 * (Y - X**2)**2 + (1.0 - X)**2

    if levels is None:
        levels = np.geomspace(1e-2, 1e4, 20)

    plt.figure(figsize=(7, 6))
    cs = plt.contour(X, Y, Z, levels=levels)
    plt.clabel(cs, inline=True, fontsize=8, fmt="%.3g")

    plt.plot(path[:, 0], path[:, 1], marker="o", linewidth=1.5)

    plt.title("Rosenbrock: Newton with line search (Task 5)")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.xlim(x_range)
    plt.ylim(y_range)
    plt.grid(True)

    if savepath:
        plt.savefig(savepath, dpi=200)
        print("Saved figure to:", savepath)

    if show:
        plt.show()
    else:
        plt.close()

    return info

if __name__ == "__main__":
    print("=== Task 1–4 sanity (Exact LS) ===")
    prob = OptProblem(rosen)  # or OptProblem(rosen, rosen_grad)
    solver = Newton(tol=1e-8, max_iter=200, line_search=ExactLineSearch())
    x0 = np.array([-1.2, 1.0])
    x_star, f_star, info = solver.optimize(prob, x0=x0)
    print("x* =", x_star, "f* =", f_star, "status:", info["status"], "iters:", info["iters"], "||g||:", info["grad_norm"])

    print("\n=== Task 5 plot (reuse info) ===")
    run_newton_and_plot(info=info, show=True)

    print("\n=== Optional: swap to Wolfe LS ===")
    prob2 = OptProblem(rosen)
    solver_wolfe = Newton(tol=1e-8, max_iter=200, line_search=WolfeLineSearch(c1=1e-4, c2=0.9))
    _ = run_newton_and_plot(solver=solver_wolfe, prob=prob2, x0=x0, show=False, savepath="rosen_wolfe.png")
