from Task1 import ClassicalNewtonMethod
import numpy as np

def func(x):
    x_1 = x[0]
    x_2 = x[1]
    return (100 * (x_2 - x_1 ** 2) ** 2 + (1 - x_1) ** 2)

def grad(x):
    x_1 = x[0]
    x_2 = x[1]
    df_dx1 = - 400 * x_1 * (x_2 - x_1 ** 2) - 2 * x_1 -2
    print(df_dx1)
    df_dx2 = 200 * (x_2 - x_1 ** 2)
    grad = np.array([df_dx1, df_dx2])
    return grad

x0 = np.array([0.5, 2])

test_run = ClassicalNewtonMethod(func, grad, x0)

# hess = test_run.approx_hess(x0)
minimizer = test_run.optimization()

# print(hess)
# print(minimizer)