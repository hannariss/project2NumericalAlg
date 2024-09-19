class OptimizationProblem:

    def __init__(self, objective_func, gradient=None):
        self.objective_func = objective_func
        self.gradient = gradient

