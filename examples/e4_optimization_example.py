"""
Goals of this part of the examples:
1. Learn how to create a custom `Optimizer` class
2. Learn the different optimizer frameworks
3. See the difference in optimization when using newton-based methods and evolutionary algorithms.
   The difference is, that newton based methods (like L-BFGS-B) are vastly faster in both convex and
   concave problems, but they are not guaranteed to find the global minimum and can get stock in local optima. 
   Evolutionary algorithms (like the genetic algorithm) are substantially slower, 
   but they can overcome local optima, as shown in the concave examples.
   
   
"""
import time
from pprint import pformat

# Start by importing all relevant packages
import matplotlib.pyplot as plt
import numpy as np

# Imports from ebcpy
from ebcpy.optimization import Optimizer

PLT_STANDARD_COLORS = ['g', 'r', 'c', 'm', 'y', 'k']

FRAMEWORK_METHODS = {
    "scipy_differential_evolution": ("best1bin", {}),
    "scipy_minimize": ("L-BFGS-B", {"x0": [0.3]}),
    "dlib_minimize": (None, {"num_function_calls": 1000}),
    "pymoo": ("ga", {"verbose": False}),
    "bayesian_optimization": (None, {"xi": 1.2,
                                     "kind_of_utility_function": "ucb"})
}

def main_loop(optimizer: Optimizer,
              plot_step: callable,
              n_vars: int):
    """
    Execute the main optimization loop for different frameworks and methods.

    This function iterates through predefined optimization frameworks and methods,
    applies them to the given optimizer, and records the execution time for each.
    It also calls a plotting function for each optimization result.

    Args:
        optimizer (Optimizer): An instance of a custom Optimizer class.
        plot_step (callable): A function to plot each optimization result.
        n_vars (int): The number of variables in the optimization problem.

    Returns:
        None
    """
    summary = {}
    for i, (framework, method_kwargs) in enumerate(FRAMEWORK_METHODS.items()):
        method, kwargs = method_kwargs
        if method == "L-BFGS-B":
            kwargs["x0"] = n_vars * [-1]
        optimizer.logger.info(f"Optimizing framework {framework} with method {method}")
        try:
            start = time.perf_counter()
            res = optimizer.optimize(framework=framework, method=method, **kwargs)
            dur = time.perf_counter() - start
            optimizer.logger.info(f"Optimization took {dur} seconds")
            summary[framework] = dur
        except ImportError as err:
            optimizer.logger.error(f"Could not optimize due to import error: {err}")
            continue
        plot_step(res, framework, method, i)
    
    optimizer.logger.info(f"Optimization times summary:\n{pformat(summary)}")
    

def concave_1d_example(with_plot=True):
    """
    Run the main optimization routine for a 1D concave problem.

    This function defines a 1D concave optimization problem, sets up the optimizer,
    and runs the optimization using various frameworks. It also plots the results
    if specified.

    Args:
        with_plot (bool, optional): Whether to display the plot. Defaults to True.

    Returns:
        None
    """
    
    class ConcaveProblemOptimizer1D(Optimizer):
        """
        Define a custom Optimizer by inheriting.
        This optimizer tries to find the minimum for the function 
        f(x) = -1*(exp(-(x - 2) ** 2) + exp(-(x - 6) ** 2 / 10) + 1/ (x ** 2 + 1)), which is an arbitrary 
        concave function.
        """
        def __init__(self, **kwargs):
            """
            Theoretically, additional data which is needed for the optimization can be passed here and 
            stored as class attributes. Not necessary for this example.
            """
            super().__init__(**kwargs)
        
        def obj(self, xk, *args):
            return -1*(np.exp(-(xk[0] - 2) ** 2) + np.exp(-(xk[0] - 6) ** 2 / 10) + 1/ (xk[0] ** 2 + 1))
        
        
    bounds = [(-2, 10)]
    x_area = np.linspace(-2, 10, 1000).reshape(1, -1)
    cpo = ConcaveProblemOptimizer1D(bounds=bounds)
    y = cpo.obj(x_area)
    plt.figure()
    plt.plot(x_area.flatten(), y)
    
    def concav_1d_plot_step(res, framework, method, i):
        plt.plot(res.x, res.fun, f"{PLT_STANDARD_COLORS[i]}o", label=f"{framework}: {method}")
        
    main_loop(optimizer=cpo, 
              plot_step=concav_1d_plot_step,
              n_vars=len(bounds))
    
    if with_plot:
        plt.title("1D Concave Problem Optimization")
        plt.xlabel("X")
        plt.ylabel("objective function value")
        plt.legend()
        plt.show()
        
def convex_1d_example(with_plot=True):
    """
    Run the main optimization routine for a 1D convex problem.

    This function defines a 1D convex optimization problem, sets up the optimizer,
    and runs the optimization using various frameworks. It also plots the results
    if specified.

    Args:
        with_plot (bool, optional): Whether to display the plot. Defaults to True.

    Returns:
        None
    """
    
    class ConvexProblemOptimizer1D(Optimizer):
        """
        Define a custom Optimizer by inheriting.
        This optimizer tries to find the minimum for the function 
        f(x) = (x - 3)**2 + 2, which is an arbitrary convex function.
        """
        def __init__(self, **kwargs):
            """
            Theoretically, additional data which is needed for the optimization can be passed here and 
            stored as class attributes. Not necessary for this example.
            """
            super().__init__(**kwargs)
        
        def obj(self, xk, *args):
            return (xk[0] - 3)**2 + 2
        
        
    bounds = [(-5, 10)]
    x_area = np.linspace(-2, 10, 1000).reshape(1, -1)
    cpo = ConvexProblemOptimizer1D(bounds=bounds)
    y = cpo.obj(x_area)
    plt.figure()
    plt.plot(x_area.flatten(), y)
    
    def convex_1d_plot_step(res, framework, method, i):
        plt.plot(res.x, res.fun, f"{PLT_STANDARD_COLORS[i]}o", label=f"{framework}: {method}")
        
    main_loop(optimizer=cpo, 
              plot_step=convex_1d_plot_step,
              n_vars=len(bounds))
    
    if with_plot:
        plt.title("1D Convex Problem Optimization")
        plt.xlabel("X")
        plt.ylabel("objective function value")
        plt.legend()
        plt.show()
        
def concave_2d_example(with_plot=True):
    """
    Run the main optimization routine for a 2D concave problem.

    This function defines a 2D concave optimization problem, sets up the optimizer,
    and runs the optimization using various frameworks. It also plots the results
    if specified.

    Args:
        with_plot (bool, optional): Whether to display the plot. Defaults to True.

    Returns:
        None
    """

    class ConcaveProblemOptimizer2D(Optimizer):
        """
        Define a custom Optimizer by inheriting.
        This optimizer tries to find the minimum for the function
        f=(x, y) = -1*(exp(-(x - 2)**2 - (y - 2)**2) + exp(-((x - 6)**2 / 10) - 
                      ((y - 6)**2 / 10)) + 1 / (x**2 + y**2 + 1)),
        which is an arbitrary concave function.
        """
        def __init__(self, **kwargs):
            """
            Theoretically, additional data which is needed for the optimization can be passed here and
            stored as class attributes. Not necessary for this example.
            """
            super().__init__(**kwargs)
        
        def obj(self, xk, *args):
            x, y = xk
            return -1 * (np.exp(-(x - 2)**2 - (y - 2)**2) + 
                         np.exp(-((x - 6)**2 / 10) - ((y - 6)**2 / 10)) + 
                         1 / (x**2 + y**2 + 1))

    bounds = [(-2, 10), (-2, 10)]
    x_area = np.linspace(-2, 10, 100)
    y_area = np.linspace(-2, 10, 100)
    X, Y = np.meshgrid(x_area, y_area)
    cpo = ConcaveProblemOptimizer2D(bounds=bounds)
    Z = np.array([cpo.obj([x, y]) for x, y in zip(X.flatten(), Y.flatten())]).reshape(X.shape)
    
    plt.figure(figsize=(12, 10))
    contour = plt.contour(X, Y, Z, levels=20)
    plt.colorbar(contour)
    
    def concav_2d_plot_step(res, framework, method, i):
        plt.plot(res.x[0], res.x[1], f"{PLT_STANDARD_COLORS[i]}o", 
                     markersize=10, label=f"{framework}: {method} (obj: {res.fun:.2f})")
    
    main_loop(optimizer=cpo, 
              plot_step=concav_2d_plot_step,
              n_vars=len(bounds))
    
    if with_plot:
        plt.title("2D Concave Problem Optimization")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.legend()
        plt.show()
        
def convex_2d_example(with_plot=True):
    """
    Run the main optimization routine for a 2D convex problem.

    This function defines a 2D convex optimization problem, sets up the optimizer,
    and runs the optimization using various frameworks. It also plots the results
    if specified.

    Args:
        with_plot (bool, optional): Whether to display the plot. Defaults to True.

    Returns:
        None
    """

    class ConvexProblemOptimizer2D(Optimizer):
        """
        Define a custom Optimizer by inheriting.
        This optimizer tries to find the minimum for the function
        f=(x, y) = (x - 2)**2 + (y - 3)**2 + x*y,
        which is an arbitrary convex function.
        """
        def __init__(self, **kwargs):
            """
            Theoretically, additional data which is needed for the optimization can be passed here and
            stored as class attributes. Not necessary for this example.
            """
            super().__init__(**kwargs)
        
        def obj(self, xk, *args):
            x, y = xk
            return (x - 2)**2 + (y - 3)**2 + x*y

    bounds = [(-5, 10), (-5, 10)]
    x_area = np.linspace(-5, 10, 100)
    y_area = np.linspace(-5, 10, 100)
    X, Y = np.meshgrid(x_area, y_area)
    cpo = ConvexProblemOptimizer2D(bounds=bounds)
    Z = np.array([cpo.obj([x, y]) for x, y in zip(X.flatten(), Y.flatten())]).reshape(X.shape)
    
    plt.figure(figsize=(12, 10))
    contour = plt.contour(X, Y, Z, levels=20)
    plt.colorbar(contour)
    
    def convex_2d_plot_step(res, framework, method, i):
        plt.plot(res.x[0], res.x[1], f"{PLT_STANDARD_COLORS[i]}o", 
                     markersize=10, label=f"{framework}: {method} (obj: {res.fun:.2f})")
    
    main_loop(optimizer=cpo, 
              plot_step=convex_2d_plot_step,
              n_vars=len(bounds))
    
    if with_plot:
        plt.title("2D Convex Problem Optimization")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.legend()
        plt.show()

def main(with_plot=True):
    convex_1d_example(with_plot=with_plot)
    concave_1d_example(with_plot=with_plot)
    convex_2d_example(with_plot=with_plot)
    concave_2d_example(with_plot=with_plot)

if __name__ == '__main__':
    main()
