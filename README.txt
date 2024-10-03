# Optimization

**Authors:** Johanna Rissbacher, Jule Grimm

## Project Structure

The project contains the following files:

- **OptimizationMethods.py**: Contains classes with methods for solving optimization problems. 
- **main_task5_task8.py**: Executes and plots the Newton and Quasi-Newton methods for the Rosenbrock function.
- **main_task11_task12.py**: Executes the BFGS method on the Chebyshev problem and investigates the quality of the Hessian approximation.
- **chebyquad_problem.py**: Contains the implementation of the Chebyshev problem.

## Class Structure (OptimizationMethods.py)

- **OptimizationProblem**
- **GeneralOptimizationMethod**
    - **ClassicalNewtonMethod**
    - **QuasiNewtonMethods**
        - **GoodBroyden**
        - **BadBroyden**
        - **SymmetricBroyden**
        - **DFP**
        - **BFGS**

## Class Descriptions

### OptimizationProblem
- **Purpose**: Takes an objective function and optionally its gradient as input.
- **Note**: This class was created to fulfill Task 1 but is not used further in the project.

### GeneralOptimizationMethod
- **Inputs**: 
  - Objective function
  - Its gradient
  - An initial guess
  - Optional parameters: tolerance, maximum iteration number, and a guess of the steepness of the function (this affects the stopping criterion).
- **Methods**:
    - `s_k`: Computes the Newton direction.
    - `exact_line_search`: Performs exact line search.
    - `inexact_line_search`: Performed with Powell-Wolfe algorithm (provided in lecture notes).
    - `x_new`: Computes the new `x` with alpha and `s`.
    - `update_hess`: Updates the Hessian approximation.
    - `residual_crit`: Residual criterion for stopping.
    - `cauchy_crit`: Cauchy criterion for stopping.

### ClassicalNewtonMethod (inherits from GeneralOptimizationMethod)
- **Methods**:
    - `approx_hess`: Computes the Hessian with finite differences.
    - `optimization_exact_ls`: Executes optimization using exact line search.
    - `optimization_inexact_ls`: Executes optimization using inexact line search.

### QuasiNewtonMethods (inherits from GeneralOptimizationMethod)
- **Methods**:
    - `s_k`: Computes the Newton direction.
    - `optimization_inexact_ls`: Executes optimization using inexact line search.

### GoodBroyden, BadBroyden, SymmetricBroyden, DFP, BFGS (all inherit from QuasiNewtonMethods)
- **Common Methods**:
    - `update_hess`: Updates the Hessian approximation.
    - `optimization_inexact_ls`: Executes optimization using inexact line search.
