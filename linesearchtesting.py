from linesearch import *


def function_f(x):
    return 100 * x**4 + (1 - x)**2

def function_deriv(x):
    return np.array([400 * x**3 - 2 * (1 - x)])

# Test case
x_init = 0.0
direction = 1.0

alpha = InexactLineSearchMethod(
    f_func= function_f,
    f_deriv= function_deriv,
    alpha_init=0.1,
    direction=direction,
    x_init=np.array([x_init])
)

print(f"Optimal step length: {alpha}")