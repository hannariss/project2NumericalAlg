class OptimizationProblem:

    def __init__(self, objective_func, gradient=None):
        self.objective_func = objective_func
        self.gradient = gradient

class GeneralOptimizationMethod:
    def __init__(self, func, grad, x0, tol=1e-5, max_iter=1000):
        """
        Initialize the optimization method.

        Parameters:
        - func: The objective function to minimize.
        - grad: The gradient of the objective function.
        - x0: Initial guess for the minimum.
        - tol: Tolerance for the stopping criterion.
        - max_iter: Maximum number of iterations.
        """
        self.func = func
        self.grad = grad
        self.x0 = x0
        self.tol = tol
        self.max_iter = max_iter