import numpy as np
import scipy.optimize as so
import chebyquad_problem as ch
import Task1

# define initial guess (x0)
#x0 = np.linspace(0, 1, 8)
x0 = np.linspace(0, 1, 4)

# define initial hess
hess = np.eye(len(x0))

test_run = Task1.BFGS(ch.chebyquad, ch.gradchebyquad, x0)
minimizer, steps = test_run.optimization_inexact_ls(sigma=0.1, rho=0.01, alpha_min=5, hess=hess)

# scipy optimization
xmin = so.fmin_bfgs(ch.chebyquad, x0, ch.gradchebyquad)

print(f'our x_min {minimizer}')
print(f'scipys min {xmin}')


# Task 12:
# With growing k the hessian approximation, which we compute in each iteration in the update hessian step,
# is converging to the exact hessian. This does not mean that the approximation of the hessian is necessarily of 
# better quality in every single step but in the long term after k steps we have a sufficient approximation of the hessian.

result = so.minimize(ch.chebyquad, x0, method='BFGS', options={'disp': True})
print(result.hess_inv)