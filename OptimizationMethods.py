import numpy as np
from scipy.optimize import minimize_scalar, line_search

class OptimizationProblem: # asked Task1 --not really used
    def __init__(self, objective_func, gradient=None):
        self.objective_func = objective_func
        self.gradient = gradient

class GeneralOptimizationMethod:
    iterations = 0
    
    def __init__(self, func, grad, x0, tol=1e-5, k=1000, steep=True):
        """
        Initialize the optimization method.

        Parameters:
        - func: Callable. The objective function to minimize.
        - grad: Callable. The gradient of the objective function.
        - x0: np.ndarray. Initial guess for the minimum.
        - tol: float, optional. Tolerance for the stopping criterion (default is 1e-5).
        - k: int, optional. Maximum number of iterations (default is 1000).
        - steep: bool, optional. Determines which stopping criteria to use (default is True). 
                 If steep is True, use residual criterion, otherwise use Cauchy criterion.
        """
        self.func = func
        self.grad = grad
        self.x0 = x0
        self.tol = tol
        self.k = k
        self.steep = steep

    ## First step of Newton Methods
    def s_k(self, x, hess):
        """
        Compute the search direction for the current iteration.

        Parameters:
        - x: np.ndarray. The current point in the optimization.
        - hess: np.ndarray. The current approximation of the Hessian matrix.

        Returns:
        - np.ndarray. The search direction (negative gradient scaled by Hessian).
        """
        return -(np.dot(np.linalg.inv(hess), self.grad(x)))  # s^(k) := - G^(x^(k))^-1 * g^(x^(k))
    
    ## Second step of Newton Methods
    def exact_line_search(self, x, s_k):
        """
        Perform an exact line search to find the optimal step size that minimizes the objective function 
        along the search direction.

        Parameters:
        - x: np.ndarray. The current point in the optimization.
        - s_k: np.ndarray. The search direction.

        Returns:
        - float. The optimal step size (alpha) found via line search.
        """
        def phi(alpha):
            return self.func(x + alpha * s_k)
        
        alpha_opt = minimize_scalar(phi)
        return alpha_opt.x # x is solution array of the optimization result from minimize_scalar
    
    def inexact_line_search(self, x, s_k, sigma, rho, alpha_min):
        """
        Perform an inexact line search using the Armijo, Wolfe, and Goldstein conditions.

        Parameters:
        - x: np.ndarray. The current point in the optimization.
        - s_k: np.ndarray. The search direction.
        - sigma: float. Parameter for the Armijo and Goldstein conditions (between 0 and 1).
        - rho: float. Parameter for the Wolfe condition (between 0 and 1).
        - alpha_min: float. Initial step size guess.

        Returns:
        - float. The step size (alpha) that satisfies both Armijo and Wolfe conditions.
        """

        # Define Armijo condition
        def armijo(x, s_k, alpha, sigma):
            if self.func(x + alpha * s_k) <= self.func(x) + sigma * alpha * np.dot(s_k, self.grad(x)):
                return True
            else:
                return False
        
        # Define Powell-Wolfe condition
        def wolfe(x, s_k, alpha, rho):
            if np.dot(s_k, self.grad(x + alpha * s_k)) >= rho * np.dot(s_k, self.grad(x)):
                return True
            else:
                return False
        
        # Define Goldstein condition
        def goldstein(x, s_k, alpha, sigma):
            if self.func(x + alpha * s_k) >= self.func(x) + (1 - sigma) * alpha * np.dot(s_k, self.grad(x)):
                return True
            else:
                return False
        
        # Adjust alpha_min to satisfy Armijo condition
        while not armijo(x, s_k, alpha_min, sigma): 
            alpha_min = alpha_min / 2
        
        alpha_max = alpha_min
       
        # Increase alpha_max to satisfy Armijo condition
        while armijo(x, s_k, alpha_max, sigma): 
            alpha_max = 2 * alpha_max
        
        # Bisection search to satisfy Wolfe condition
        while not wolfe(x, s_k, alpha_min, rho): 
            alpha_0 = (alpha_min + alpha_max)/2
            if armijo(x, s_k, alpha_0, sigma):
                alpha_min = alpha_0
            else:
                alpha_max = alpha_0

        return alpha_min
    
    # Third step of Newton Methods
    def x_new(self, x, s, alpha):
        """
        Compute the new point based on the current point, search direction, and step size.

        Parameters:
        - x: np.ndarray. The current point in the optimization.
        - s: np.ndarray. The search direction.
        - alpha: float. The step size.

        Returns:
        - np.ndarray. The updated point for the next iteration.
        """
        return x + (alpha * s)
    
    # Fourth step of Quasi-Newton Methods
    def update_hess(self):
        """
        Update the Hessian approximation. 
        This method should be implemented by specific Quasi-Newton methods (like BFGS, DFP, etc.).
        
        Returns:
        - None.
        """
        return None
    
    ## Stopping criteria for optimization
    def residual_crit(self, x):
        """
        Check if the optimization process should stop based on the residual criterion.

        Parameters:
        - x: np.ndarray. The current point in the optimization.

        Returns:
        - bool. True if the residual is below the tolerance or maximum iterations have been reached, False otherwise.
        """
        self.iterations += 1
        criterion = False
        residual = np.linalg.norm(self.grad(x))
        if residual < self.tol:
            criterion = True
        if self.iterations > self.k:
            print("Maximum iterations reached.")
            criterion = True
        return criterion
    
    def cauchy_crit(self, x, x_new):
        """
        Check if the optimization process should stop based on the Cauchy criterion.

        Parameters:
        - x: np.ndarray. The current point in the optimization.
        - x_new: np.ndarray. The newly computed point after an iteration.

        Returns:
        - bool. True if the difference between x and x_new is below the tolerance, False otherwise.
        """
        criterion = False
        cauchy = np.linalg.norm((x_new-x))
        if cauchy < self.tol:
            criterion = True
        return criterion

class ClassicalNewtonMethod(GeneralOptimizationMethod):
    def approx_hess(self, x, h=1e-5):
        """
        Approximate the Hessian matrix using finite differences.

        Parameters:
        - x: np.ndarray. The current point in the optimization.
        - h: float, optional. The step size for finite difference approximation (default is 1e-5).

        Returns:
        - np.ndarray. Symmetric approximation of the Hessian matrix.
        """
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

    def optimization_exact_ls(self, x0=None):
        """
        Perform optimization using Newton's method with exact line search.

        Parameters:
        - x0: np.ndarray, optional. The starting point for optimization (if not provided, self.x0 is used).

        Returns:
        - np.ndarray. The final optimized point.
        - list. A list of points representing the optimization steps.
        """
        self.x0 = x0 if x0 is not None else self.x0  # Use x0 from input or default to self.x0
        x = self.x0
        steps = []

        # Optimization loop with residual criterion (for steep functions)
        if self.steep: # by default function is defined as steep
            while not self.residual_crit(x):
                hess = self.approx_hess(x) # Approximate the Hessian
                s = self.s_k(x, hess) # Compute the search direction
                alpha = self.exact_line_search(x, s)  # Compute step size (alpha)
                x = x_new # Update x
                steps.append(x)
        else: # For non-steep functions, use Cauchy criterion
            while True:
                hess = self.approx_hess(x)
                s = self.s_k(x, hess)
                alpha = self.exact_line_search(x, s) 
                x_new = self.x_new(x, s, alpha)
                if self.cauchy_crit(x, x_new):
                    break
                x = x_new
                steps.append(x)
        return x_new, steps

    def optimization_inexact_ls(self, sigma, rho, alpha_min, x0=None):
        """
        Perform optimization using Newton's method with inexact line search (Armijo-Wolfe conditions).

        Parameters:
        - sigma: float. Armijo condition parameter (between 0 and 1).
        - rho: float. Wolfe condition parameter (between 0 and 1).
        - alpha_min: float. Initial step size guess.
        - x0: np.ndarray, optional. The starting point for optimization (if not provided, self.x0 is used).

        Returns:
        - np.ndarray. The final optimized point.
        - list. A list of points representing the optimization steps.
        """
        self.x0 = x0 if x0 is not None else self.x0  # Use x0 from input or default to self.x0
        x = self.x0
        steps = []

        # Optimization loop with residual criterion (for steep functions)
        if self.steep: # by default function is defined as steep
            while not self.residual_crit(x):
                hess = self.approx_hess(x) # Approximate the Hessian
                s = self.s_k(x, hess) # Compute the search direction
                alpha = self.inexact_line_search(x, s, sigma, rho, alpha_min) # Compute step size (alpha)
                x_new = self.x_new(x, s, alpha) # Update x
                x = x_new
                steps.append(x)
        else: # For non-steep functions, use Cauchy criterion
            while True:
                hess = self.approx_hess(x)
                s = self.s_k(x, hess)
                alpha = self.inexact_line_search(x, s, sigma, rho, alpha_min) 
                x_new = self.x_new(x, s, alpha)
                if self.cauchy_crit(x, x_new):
                    break
                x = x_new
                steps.append(x)
        return x_new, steps

class QuasiNewtonMethods(GeneralOptimizationMethod):
    def s_k(self, x, hess):
        """
        Compute the search direction for Quasi-Newton methods.

        Parameters:
        - x: np.ndarray. The current point in the optimization.
        - hess: np.ndarray. The approximation of the Hessian.

        Returns:
        - np.ndarray. The search direction s_k.
        """
        return (np.dot(-hess, self.grad(x))) 
    
    def optimization_inexact_ls(self, sigma, rho, alpha_min, hess, x0=None):
        """
        Perform optimization using Quasi-Newton methods with inexact line search.

        Parameters:
        - sigma: float. Armijo condition parameter.
        - rho: float. Wolfe condition parameter.
        - alpha_min: float. Initial step size guess.
        - hess: np.ndarray. The Hessian matrix or its approximation.
        - x0: np.ndarray, optional. The starting point for optimization.

        Returns:
        - np.ndarray. The final optimized point.
        - list. A list of points representing the optimization steps.
        """
        self.x0 = x0 if x0 is not None else self.x0  # Use x0 from input or default to self.x0
        x = self.x0
        steps = []

        # Optimization loop with residual criterion (for steep functions)
        if self.steep: 
            while not self.residual_crit(x):
                s = self.s_k(x, hess) # 1) Compute Search direction (s)
                alpha = self.inexact_line_search(x, s, sigma, rho, alpha_min) # 2) Compute stepsize (alpha) with linesearch
                if self.func(x + alpha*s) >= self.func(x):
                    print("Warning: s_k is not a descent direction.")
                x_new = self.x_new(x, s, alpha) # 3) Update x
                hess = self.update_hess(x, x_new, hess) # 4) Update hessian
                print(hess)
                x = x_new
                steps.append(x)
        else:  # For non-steep functions, use Cauchy criterion
            while True:
                s = self.s_k(x, hess)
                alpha = self.inexact_line_search(x, s, sigma, rho, alpha_min)
                x_new = self.x_new(x, s, alpha)
                if self.cauchy_crit(x, x_new):
                    break
                hess = self.update_hess(x, x_new, hess) 
                x = x_new
                steps.append(x)       
        return x_new, steps


class GoodBroyden(QuasiNewtonMethods):
    def update_hess(self, x, x_new, hess):
        """
        Update the Hessian matrix using Good Broyden's method.

        Parameters:
        - x: np.ndarray. The current point.
        - x_new: np.ndarray. The new point after step.
        - hess: np.ndarray. The current Hessian matrix.

        Returns:
        - np.ndarray. The updated Hessian matrix.
        """
        delta = x_new - x # result: vector
        gamma = self.grad(x_new) - self.grad(x) # result: vector

        term1 = delta - np.dot(hess, gamma)
        term2 = np.dot(np.dot(delta.T, hess), gamma)

        # Update Hessian
        hess_new = hess + np.outer(term1, np.dot(delta.T, hess)) / term2
        hess_new = 0.5 * (hess_new + hess_new.T)

        return hess_new
    
    def optimization_inexact_ls(self, sigma, rho, alpha_min, hess, x0=None):
        """
        Perform optimization using Good Broyden's method with inexact line search.
        """
        return super().optimization_inexact_ls(sigma, rho, alpha_min, hess, x0)


class BadBroyden(QuasiNewtonMethods):
    def update_hess(self, x, x_new, hess):
        """
        Update the Hessian matrix using Bad Broyden's method.

        Parameters:
        - x: np.ndarray. The current point.
        - x_new: np.ndarray. The new point after step.
        - hess: np.ndarray. The current Hessian matrix.

        Returns:
        - np.ndarray. The updated Hessian matrix.
        """
        delta = x_new - x
        gamma = self.grad(x_new) - self.grad(x)

        term1 = np.dot(gamma.T, gamma)
        if abs(term1) < 1e-8:  # Ensure no division by near-zero
            term1 = np.sign(term1) * 1e-8
        
        term2 = (delta - np.dot(hess, gamma)) / term1
        
        # Update Hessian
        hess_new = hess + np.outer(term2, gamma.T)
        hess_new = 0.5 * (hess_new + hess_new.T)
        return hess_new
    
    def optimization_inexact_ls(self, sigma, rho, alpha_min, hess, x0=None):
        """
        Perform optimization using Bad Broyden's method with inexact line search.
        """
        return super().optimization_inexact_ls(sigma, rho, alpha_min, hess, x0)


class SymmetricBroyden(QuasiNewtonMethods):
    def update_hess(self, x, x_new, hess):
        """
        Update the Hessian matrix using Symmetric Broyden's method.

        Parameters:
        - x: np.ndarray. The current point.
        - x_new: np.ndarray. The new point after step.
        - hess: np.ndarray. The current Hessian matrix.

        Returns:
        - np.ndarray. The updated Hessian matrix.
        """
        delta = x_new - x # result: vector
        gamma = self.grad(x_new) - self.grad(x) # result: vector
        u = delta - np.dot(hess, gamma)
        
        a_denominator = np.dot(u.T, gamma)
        if abs(a_denominator) < 1e-8:  # Ensure no division by near-zero
            a_denominator = np.sign(a_denominator) * 1e-8
        
        a = 1 / a_denominator
        
        # Update Hessian
        hess_new = hess + a * np.outer(u, u)
        return hess_new
    
    def optimization_inexact_ls(self, sigma, rho, alpha_min, hess, x0=None):
        """
        Perform optimization using Symmetric Broyden's method with inexact line search.
        """
        return super().optimization_inexact_ls(sigma, rho, alpha_min, hess, x0)


class DFP(QuasiNewtonMethods):
    def update_hess(self, x, x_new, hess):
        """
        Update the Hessian matrix using Davidon-Fletcher-Powell (DFP) method.

        Parameters:
        - x: np.ndarray. The current point.
        - x_new: np.ndarray. The new point after step.
        - hess: np.ndarray. The current Hessian matrix.

        Returns:
        - np.ndarray. The updated Hessian matrix.
        """
        delta = x_new - x # Result: vector
        gamma = self.grad(x_new) - self.grad(x) # Result: vector

        # Compute terms for the DFP update
        delta_gamma = np.dot(delta.T, gamma)
        gamma_hess_gamma = np.linalg.multi_dot([gamma.T, hess, gamma])

        # Update Hessian
        term1 = np.outer(delta, delta) / delta_gamma
        term2 = np.linalg.multi_dot([hess, np.outer(gamma, gamma), hess]) / gamma_hess_gamma

        hess_new = hess + term1 - term2
        hess_new = 0.5 * (hess_new + hess_new.T)
        return hess_new
    
    def optimization_inexact_ls(self, sigma, rho, alpha_min, hess, x0=None):
        """
        Perform optimization using DFP method with inexact line search.
        """
        return super().optimization_inexact_ls(sigma, rho, alpha_min, hess, x0)


class BFGS(QuasiNewtonMethods):
    def update_hess(self, x, x_new, hess):
        """
        Update the Hessian matrix using Broyden-Fletcher-Goldfarb-Shanno (BFGS) method.

        Parameters:
        - x: np.ndarray. The current point.
        - x_new: np.ndarray. The new point after step.
        - hess: np.ndarray. The current Hessian matrix.

        Returns:
        - np.ndarray. The updated Hessian matrix.
        """
        delta = x_new - x # result: vector
        gamma = self.grad(x_new) - self.grad(x) # result: vector

        # Compute denominator
        delta_gamma = np.dot(delta.T, gamma) # calculates the denominator used in calculation

        term1 = (1 + np.dot(gamma.T, np.dot(hess, gamma)) / delta_gamma) * np.outer(delta, delta) / delta_gamma
        term2 = np.outer(np.dot(hess, gamma), delta) / delta_gamma
        term3 = np.outer(delta, np.dot(gamma.T, hess)) / delta_gamma

        # Update the Hessian
        hess_new = hess + term1 - term2 - term3
        hess_new = 0.5 * (hess_new + hess_new.T) 
        return hess_new

    def optimization_inexact_ls(self, sigma, rho, alpha_min, hess, x0=None):
        """
        Perform optimization using BFGS method with inexact line search.
        """
        return super().optimization_inexact_ls(sigma, rho, alpha_min, hess, x0)