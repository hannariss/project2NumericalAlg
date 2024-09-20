import numpy as np

class OptimizationProblem:
    def __init__(self, objective_func, gradient=None):
        self.objective_func = objective_func
        self.gradient = gradient

class GeneralOptimizationMethod:
    def __init__(self, func, grad, x0, tol=1e-5, k=1000):
        """
        Initialize the optimization method.

        Parameters:
        - func: The objective function to minimize.
        - grad: The gradient of the objective function.
        - x0: Initial guess for the minimum.
        - tol: Tolerance for the stopping criterion.
        - k: Maximum number of iterations.
        """
        self.func = func
        self.grad = grad
        self.x0 = x0
        self.tol = tol
        self.k = k

    # First step of Quasi-Newton Methods
    def compute_direction(self, hess_approx):
        return -hess_approx * self.grad  # s^(k) := - H^(k) * g^(k)
    
    # Second step of Quasi-Newton Methods
    def line_search(self):
        return None # Note: Check that alpha exists?
    
    # Third step of Quasi-Newton Methods
    def newton_step(self, x, alpha, s):
        return x + alpha * s  # x^(k+1) = x^(k) + alpha^(k) * s^(k)
    
    # Third step of Quasi-Newton Methods
    def update_hess(self):
        return None
    

class ClassicalNewtonMethod(GeneralOptimizationMethod):
    def compute_direction(self, hess):
        return (-1/hess)*self.grad  # s^(k) := - G^(x^(k))^-1 * g^(x^(k))
    
    def newton_step(self, x, s):
        return x + s

    def approx_hess(self, x, h=1e-5):
        n = len(x) # Dimension of x
        hess = np.zeros((n, n)) # Creates n x n Matrix with zeros

        for i in range(n-1): # Range is 0 - n-1 because normally we would start at ix 1
            for j in range(n-1):
                # Create Unit vector in i-th and j-th direction
                u_i = np.zeros(n) 
                u_j = np.zeros(n)
                u_i[i] = 1
                u_j[j] = 1
                
                ## Compute hessian approximation
                # Compute single components of formula
                f_ij = self.func(x + h * u_i + h * u_j)
                f_i = self.func(x + h * u_i)
                f_j = self.func(x + h * u_j)
                f_x = self.func(x)

                # Compute ij entry in hessian approx matrix by combining the predefined functions
                hess[i, j] = (f_ij - f_i - f_j + f_x) / (h ** 2)

        # Symmetrize the Hessian approximation matrix
        hess_sym = 1/2 * (hess + hess.T)  

        return hess_sym
    