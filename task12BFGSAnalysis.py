import numpy as np
import matplotlib.pyplot as plt
from optProblem import OptProblem

# Matrix differences
def matrix_norm1(A, B):
    A = A / np.max(A)
    B = B / np.max(B)
    diff = np.abs(B-A)
    return np.sqrt(np.sum(diff.flatten()**2))
def matrix_norm2(A, B):
    A = A / np.linalg.det(A)
    B = B / np.linalg.det(B)
    diff = np.abs(B-A)
    return np.sqrt(np.sum(diff.flatten()**2))

# test function
def myFunction(x):
    return (x[0]+3)**4 + x[1]**2
# gradient
def myGrad(x):
    return np.array([4*(x[0]+3)**3, 2*x[1]])
# hessian
def myHessian(x):
    return np.array([[12*(x[0]+3)**2, 0], [0, 2]], dtype=np.float64)
    

solver = OptProblem(myFunction)

# Run the demo bfgs and analyze the similarity between
# the inverse hessian approximation in it and numerically
termination_condition = solver.OptMethod.residual_criterion
inexact_search = solver.OptMethod.create_inexact_linesearch(solver.function, alpha_init=1)
x0 = np.array([-5, 2.])
minimum, Hk_invs, points = solver.bfgs_demo(x0, termination_criterion=termination_condition, alpha=0.5)

print(f"Found minimum: {minimum}")
# Real hessian inverses
real_Hk_invs = [np.linalg.inv(myHessian(pt)) for pt in points]
# Differences between the Hks and the real hessian
matrix_diffs = np.array([matrix_norm2(B, A) for A, B in zip(Hk_invs, real_Hk_invs)])
#print(matrix_diffs)
#print(points[-1])
print("First BFGS Hessian inverse:")
print(Hk_invs[0])
print("First real Hessian inverse:")
print(real_Hk_invs[0])
print("Normalized difference between them:")
print(matrix_norm2(Hk_invs[0], real_Hk_invs[0]))
print("Final BFGS Hessian inverse:")
print(Hk_invs[-1])
print("Final real Hessian inverse:")
print(real_Hk_invs[-1])
print("Normalized difference between them:")
print(matrix_norm2(Hk_invs[-1], real_Hk_invs[-1]))
# Plot the differences as a function of iterations
plt.plot(np.arange(len(matrix_diffs)), matrix_diffs)
plt.title(r"Difference between BFGS and analytic inverse Hessian for $(x+3)^4 + y^2$")
plt.xlabel("Iterations")
plt.ylabel("Matrix difference")
plt.show()