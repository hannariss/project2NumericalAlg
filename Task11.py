import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as so

import chebyquad_problem as ch
import Task1


# define initial guess (x0)
x0 = np.array([2, 3, 5, 3])
# x0 = np.linspace(0, 1, 8)


# define initial hess
hess = np.eye(4)

test_run = Task1.BFGS(ch.chebyquad, ch.gradchebyquad, x0, k=1000, tol=1e-5, steep=True)
minimizer, steps = test_run.optimization_inexact_ls(sigma=0.1, rho=0.01, alpha_min=5, hess=hess)

# scipy optimization
xmin= so.fmin_bfgs(ch.chebyquad, x0, ch.gradchebyquad)

# print(steps, flush=True)
print(f'our x_min {minimizer}')
print(f'scipys min {xmin}')

