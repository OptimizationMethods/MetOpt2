"""
Methods for tasks
"""

import time

import numpy as np
from numpy.typing import NDArray

from functools import wraps

from scipy.optimize import minimize

from typing import Literal

import warnings

warnings.filterwarnings("ignore")


def enhanced_time_profiler(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        wrapper.calls += 1
        start_time = time.time()
        result = func(*args, **kwargs)
        elapsed_time = time.time() - start_time
        wrapper.total_time += elapsed_time
        print(f"{func.__name__} executed in {elapsed_time:.6f} seconds.")
        if wrapper.calls > 1:
            print(
                f"Average execution time: {wrapper.total_time / wrapper.calls:.6f} seconds over {wrapper.calls} calls.")
        return result

    wrapper.calls = 0
    wrapper.total_time = 0
    return wrapper


@enhanced_time_profiler
def newton_method(f, grad, hess, x0, tol=1e-10, max_iter=1000):
    x = np.array(x0, dtype=float)
    for i in range(max_iter):
        grad_val = grad(x)
        hess_val = hess(x)
        delta_x = np.linalg.solve(hess_val, -grad_val)
        x += delta_x
        if np.linalg.norm(delta_x) < tol:
            return x, i + 1
    return x, max_iter


def newton_with_dichotomy_search_2d(f, grad_f, hessian_f, x0, tol=1e-6, max_iter=100, a=0.0, b=1.0, eps=1e-6):
    def dichotomy_search(phi, a, b, eps):
        while (b - a) / 2 > eps:
            mid = (a + b) / 2
            if phi(mid - eps) < phi(mid + eps):
                b = mid
            else:
                a = mid
        return (a + b) / 2

    x = np.array(x0, dtype=float)
    for _ in range(max_iter):
        fx = f(x)
        if np.linalg.norm(fx) < tol:
            return x, max_iter
        grad_fx = grad_f(x)
        hessian_fx = hessian_f(x)
        if np.linalg.det(hessian_fx) == 0:
            raise ValueError("Гессиан вырожден. Метод Ньютона не применим.")

        hessian_inv = np.linalg.inv(hessian_fx)

        direction = -hessian_inv @ grad_fx

        phi = lambda lmbd: np.linalg.norm(f(x + lmbd * direction)) ** 2

        lambda_opt = dichotomy_search(phi, a, b, eps)

        x = x + lambda_opt * direction

    return x, max_iter


@enhanced_time_profiler
def scipy_methods(method_type: Literal["Newton-CG", "BFGS"], point: NDArray, function, hessian, jac=None,
                  grad_func=None):
    result = minimize(fun=function, x0=point, method=method_type, jac=jac or grad_func, hess=hessian)
    return result.x, result.nit
