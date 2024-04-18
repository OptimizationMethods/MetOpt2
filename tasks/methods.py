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


def dichotomy_search(f, X, d, low, high, tol=1e-5):
	while high - low > tol:
		mid1 = low + (high - low) / 3
		mid2 = high - (high - low) / 3
		if f(X - mid1 * d) < f(X - mid2 * d):
			high = mid2
		else:
			low = mid1
	return (low + high) / 2


@enhanced_time_profiler
def newton_one_dim_search(f, grad_f, hessian_f, X0, tol=1e-6, max_iter=1000):
	X = np.array(X0, dtype=float)
	for i in range(max_iter):
		grad = grad_f(X)
		Hessian = hessian_f(X)
		H_inv = np.linalg.inv(Hessian)
		direction = -H_inv @ grad
		alpha = dichotomy_search(f, X, direction, 0, 1, tol=1e-5)
		X_new = X + alpha * direction
		if np.linalg.norm(X_new - X) < tol:
			return X_new, i + 1
		X = X_new
	return X, max_iter


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


@enhanced_time_profiler
def scipy_methods(method_type: Literal["Newton-CG", "BFGS"], point: NDArray, function, hessian, jac=None,
                  grad_func=None):
	result = minimize(fun=function, x0=point, method=method_type, jac=jac or grad_func, hess=hessian)
	return result.x, result.nit
