import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as so

import chebyquad_problem as ch
import Task1


# define initial guess (x0)
x0 = np.array([0, 1, 5])
#x0 = np.linspace(0,1,3)

# define initial hess
hess = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

test_run = Task1.BFGS(ch.chebyquad, ch.gradchebyquad, x0, k=500)
minimizer, steps = test_run.optimization_inexact_ls(sigma=10**(-2), rho=0.9, alpha_min=5, hess=hess)

xmin= so.fmin_bfgs(ch.chebyquad, x0, ch.gradchebyquad)

print(steps, flush=True)
print(f'our x_min {minimizer}')
print(f'scipys min {xmin}')


