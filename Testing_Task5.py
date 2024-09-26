import Task1
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt

def func(x):
    x_1 = x[0]
    x_2 = x[1]
    return (100 * (x_2 - x_1 ** 2) ** 2 + (1 - x_1) ** 2)

def grad(x):
    # Define symbols for x_1 and x_2 without values
    x1, x2= sp.symbols('x1 x2')
   
    # Define the function f(x_1, x_2)
    f = 100 * (x2 - x1 ** 2) ** 2 + (1 - x1) ** 2

    df_dx1 = sp.diff(f, x1)  # Partial derivative of f with respect to x_1
    df_dx2 = sp.diff(f, x2)  # Partial derivative of f with respect to x_2
    
    # Calculate partial derivatives with given values
    def calc_value(x, df_dx1, df_dx2):
        x_1 = x[0]
        x_2 = x[1]
        df_dx1 = df_dx1.subs({x1: x_1, x2: x_2}) # Fill derivative with values
        df_dx2 = df_dx2.subs({x1: x_1, x2: x_2})
        df_dx1 = float(df_dx1)
        df_dx2 = float(df_dx2)
        return np.array([df_dx1, df_dx2])
    return calc_value(x, df_dx1, df_dx2)

x0 = np.array([0, -0.5])

# # testing Newton Method
# test_run = Task1.ClassicalNewtonMethod(func, grad, x0, tol=1e-5, k=500)
# minimizer, steps = test_run.optimization_inexact_ls(sigma=10**(-2), rho=0.9, alpha_min=5)


## testing Quasi Newton Methods

# testing Good Broyden
#hess = np.array([[1, 0], [0, 1]]) # test with unit matrix 

test_run_0 = Task1.ClassicalNewtonMethod(func, grad, x0, tol=1e-5, k=500) # test with hess approx
hess_test = np.linalg.inv(test_run_0.approx_hess(x0))

test_run = Task1.SymmetricBroyden(func, grad, x0, tol=1, k=100)
minimizer, steps = test_run.optimization_inexact_ls(sigma=10**(-2), rho=0.9, alpha_min=3, hess=hess_test)

print(f'minimizer:{minimizer}')
print(steps)



#plot function
x_1 = np.linspace(-0.5, 2, 1000)
x_2 = np.linspace(-1.5, 4, 1000)
x_1, x_2 = np.meshgrid(x_1, x_2) #return a tuple of coordinate matrices

#customize shown contours
low_levels = np.linspace(0, 10, 10)
higher_levels = np.linspace(10, 800, 8)
custom_levels = np.unique(np.concatenate([low_levels, higher_levels]))

fig = plt.figure()
ax = fig.add_subplot(111)

ax.contour(x_1, x_2, func([x_1, x_2]), levels=custom_levels, colors='black', linewidths=0.5)
for points in steps:
    ax.scatter(points[0], points[1], c='orange')
ax.scatter(x0[0], x0[1], c='orange')
ax.set_xlabel('x_1')
ax.set_ylabel('x_2')
ax.set_title('Rosenbrock Function: $f(x_1, x_2) = 100 * (x_2 - x_1^2)^2 + (1 - x_1)^2$')

plt.show()








