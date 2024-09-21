from Task1 import ClassicalNewtonMethod
import numpy as np
import sympy as sp

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

test_run = ClassicalNewtonMethod(func, grad, x0, tol=1e-1, k=20000)
minimizer = test_run.optimization()

test_run = ClassicalNewtonMethod(func, grad, x0, k=200000)
minimizer_with_ls = test_run.optimization_exact_ls()

print(minimizer, grad(minimizer))
print(minimizer_with_ls, grad(minimizer_with_ls))
