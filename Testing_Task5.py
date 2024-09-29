# In this script:
# Task 5, 7 -> WORKS!
# Additional testing of Quasi Newton Method on Rosenbrock function -> except for BFGS all of them run into the divide by zero error
# BFGS gets stuck very soon

import Task1
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt

def func(x):
    """
    Define Rosenbrock function
    """

    x_1 = x[0]
    x_2 = x[1]
    return (100 * (x_2 - x_1 ** 2) ** 2 + (1 - x_1) ** 2)

def grad(x):
    """
    Define gradient of Rosenbrock function
    """

    # Define symbols for x_1 and x_2 without values
    x1, x2= sp.symbols('x1 x2')
   
    # Define the function f(x_1, x_2)
    f = 100 * (x2 - x1 ** 2) ** 2 + (1 - x1) ** 2

    df_dx1 = sp.diff(f, x1)  # Partial derivative of f with respect to x_1
    df_dx2 = sp.diff(f, x2)  # Partial derivative of f with respect to x_2
    
    # Calculate partial derivatives with given values
    def calc_value(x, df_dx1, df_dx2):
        """
        This function actually calculates the gradient with the given vector x
        """
        # unpack x vector
        x_1 = x[0]
        x_2 = x[1]

        # Fill derivative with values
        df_dx1 = df_dx1.subs({x1: x_1, x2: x_2}) 
        df_dx2 = df_dx2.subs({x1: x_1, x2: x_2})

        # convert to float values
        df_dx1 = float(df_dx1)  
        df_dx2 = float(df_dx2)
        return np.array([df_dx1, df_dx2])
    return calc_value(x, df_dx1, df_dx2)


## Define initial x vector
x0 = np.array([0, -0.5])


#------------------------------------
# Testing of Classical Newton Method
#------------------------------------

test_run = Task1.ClassicalNewtonMethod(func, grad, x0, tol=1e-5, k=500) # object gets created

# Task 5: test Classical Newton Method with EXACT line search on the Rosenbrock function
#minimizer, steps = test_run.optimization_exact_ls(x0)

# Task 7: test Classical Newton Method with INEXACT search on the Rosenbrock function
# Parameters from book:
sigma = 0.1
rho = 0.01
alpha_min = 0.1
#minimizer, steps = test_run.optimization_inexact_ls(sigma, rho, alpha_min)


#------------------------------
# Testing Quasi Newton Methods
#------------------------------

# Initial hess: Identity matrix
hess_id = np.eye(2)

# Initial hess: hess approx from Classical Newton method (approximination by finite differences)
test_run_0 = Task1.ClassicalNewtonMethod(func, grad, x0, tol=1e-5, k=500) # test with hess approx
hess = np.linalg.inv(test_run_0.approx_hess(x0))

# Bad Broyden -> divide by zero error
#test_run = Task1.BadBroyden(func, grad, x0, tol=1e-8, k=100, steep=True)

# Good Broyden -> divide by zero error
#test_run = Task1.GoodBroyden(func, grad, x0, tol=1e-8, k=100, steep=True)

# Symmetric Broyden -> divide by zero error
#test_run = Task1.SymmetricBroyden(func, grad, x0, tol=1e-8, k=100, steep=True)

# DFP -> divide by zero error
#test_run = Task1.DFP(func, grad, x0, tol=1e-8, k=100, steep=True)

# BFGS -> gets stuck very soon!
test_run = Task1.BFGS(func, grad, x0, tol=1e-8, k=20, steep=True)

minimizer, steps = test_run.optimization_inexact_ls(sigma=0.1, rho=0.01, alpha_min=0.1, hess=hess_id)

# print output
print(f'minimizer:{minimizer}')
print(steps)


#--------------------
# plot function
#--------------------

# define scale of axis
x_1 = np.linspace(-0.5, 2, 1000)
x_2 = np.linspace(-1.5, 4, 1000)
x_1, x_2 = np.meshgrid(x_1, x_2) #return a tuple of coordinate matrices

#customize shown contours (show more lines in higher levels)
low_levels = np.linspace(0, 10, 10)
higher_levels = np.linspace(10, 800, 8)
custom_levels = np.unique(np.concatenate([low_levels, higher_levels]))

fig = plt.figure()
ax = fig.add_subplot(111)

# plot contour lines
ax.contour(x_1, x_2, func([x_1, x_2]), levels=custom_levels, colors='black', linewidths=0.5)

# plot 'steps' of algorithm
for points in steps:
    ax.scatter(points[0], points[1], c='orange')

# plot initial x values
ax.scatter(x0[0], x0[1], c='orange')

# define labels and title
ax.set_xlabel('x_1')
ax.set_ylabel('x_2')
ax.set_title('Rosenbrock Function: $f(x_1, x_2) = 100 * (x_2 - x_1^2)^2 + (1 - x_1)^2$')

plt.show()








