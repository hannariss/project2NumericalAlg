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