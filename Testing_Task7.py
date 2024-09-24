from Task1 import ClassicalNewtonMethod
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

x = np.array([1, 0])

test_ls = ClassicalNewtonMethod(func, grad, x)
hess = test_ls.approx_hess(x)
s_k = -np.linalg.inv(hess).dot(grad(x)) 

# Define values as in lecture slides 
sigma = 10**(-2)
rho = 0.9
alpha_min = 5

alpha = test_ls.inexact_line_search(x, s_k, sigma, rho, alpha_min)

print(s_k)
print(alpha)