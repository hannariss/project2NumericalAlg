import numpy as np
from scipy.optimize import minimize_scalar

class OptimizationProblem:
    def __init__(self, objective_func, gradient=None):
        self.objective_func = objective_func
        self.gradient = gradient

class GeneralOptimizationMethod:
    counter = 1
    def __init__(self, func, grad, x0, tol=1e-5, k=1000, steep=True): #constructor
        """
        Initialize the optimization method.

        Parameters:
        - func: The objective function to minimize.
        - grad: The gradient of the objective function.
        - x0: Initial guess for the minimum.
        - tol: Tolerance for the stopping criterion.
        - k: Maximum number of iterations.
        - steep: refers to steepness of function, defines which stopping criteria will be used (steep -> residual, not steep -> cauchy)
        """
        self.func = func
        self.grad = grad
        self.x0 = x0
        self.tol = tol
        self.k = k
        self.steep = steep

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

        for i in range(n): 
            for j in range(n):
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
                entry = (f_ij - f_i - f_j + f_x) / (h ** 2)
                hess[i, j] = entry

        # Symmetrize the Hessian approximation matrix
        hess_sym = 1/2 * (hess + hess.T)  

        return hess_sym
    
    def residual_crit(self, x):
        self.counter += 1
        criterion = False
        residual = np.linalg.norm(self.grad(x))
        if residual < self.tol:
            criterion = True
        if self.counter > self.k:
            print("hello")
            criterion = True
        return criterion
    
    def cauchy_crit(self, x, x_new):
        criterion = False
        cauchy = np.linalg.norm((x_new-x))
        if cauchy < self.tol:
            criterion = True
        return criterion
    
    def optimization(self, x0=None):
        self.x0 = x0 if x0 is not None else self.x0  #by default x0 is defined in constructor, can be redefined in this function optionally
        x = self.x0

        if self.steep: #by default function is defined as steep
            while not self.residual_crit(x):
                hess = self.approx_hess(x)
                if np.linalg.det(hess) != 0.0:
                    s = -np.linalg.inv(hess).dot(self.grad(x)) 
                else:
                    s = -hess.dot(self.grad(x))
                x_new = x + s
                x = x_new
        else:  
            while True:
                hess = self.approx_hess(x)
                if np.linalg.det(hess) != 0.0:
                    s = -np.linalg.inv(hess).dot(self.grad(x)) 
                else:
                    s = -hess.dot(self.grad(x))
                x_new = x + s
                if self.cauchy_crit(x, x_new):
                    break
                x = x_new
        return x_new

    def exact_line_search(self, x, s_k):
        def phi(alpha):
            return self.func(x + alpha * s_k)
        alpha_opt = minimize_scalar(phi)
        return alpha_opt.x # x is solution array of the optimization result from minimize_scalar
    
    def optimization_exact_ls(self, x0=None):
        self.x0 = x0 if x0 is not None else self.x0  # by default x0 is defined in constructor, can be redefined in this function optionally
        x = self.x0

        if self.steep: # by default function is defined as steep
            while not self.residual_crit(x):
                hess = self.approx_hess(x)
                if np.linalg.det(hess) != 0.0:
                    s = -np.linalg.inv(hess).dot(self.grad(x)) 
                else:
                    s = -hess.dot(self.grad(x))
                alpha = self.exact_line_search(x, s)  # calculate alpha
                x_new = x + alpha * s
                x = x_new
        else:  
            while True:
                hess = self.approx_hess(x)
                if np.linalg.det(hess) != 0.0:
                    s = -np.linalg.inv(hess).dot(self.grad(x)) 
                else:
                    s = -hess.dot(self.grad(x))
                alpha = self.exact_line_search(x, s)  # calculate alpha
                x_new = x + alpha * s
                if self.cauchy_crit(x, x_new):
                    break
                x = x_new
        return x_new



