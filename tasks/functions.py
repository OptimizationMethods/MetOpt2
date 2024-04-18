"""
Rosenbrock type function and non-poly Himmelblau function
"""

import numpy as np
from numpy.typing import NDArray
from numbers import Real


def rosenbrock(X, a=1, b=100):
	x, y = X
	return (a - x) ** 2 + b * (y - x ** 2) ** 2


def grad_rosenbrock(X, a=1, b=100):
	x, y = X
	return np.array([-2 * (a - x) - 4 * b * x * (y - x ** 2), 2 * b * (y - x ** 2)])


def hessian_rosenbrock(X, a=1, b=100):
	x, y = X
	return np.array([[2 - 4 * b * y + 12 * b * x ** 2, -4 * b * x], [-4 * b * x, 2 * b]])


def himmelblau(X: NDArray) -> Real:
	x, y = X
	return (x ** 2 + y - 11) ** 2 + (x + y ** 2 - 7) ** 2


def grad_himmelblau(X):
	x, y = X
	return np.array(
		[4 * x * (x ** 2 + y - 11) + 2 * (x + y ** 2 - 7), 2 * (x ** 2 + y - 11) + 4 * y * (x + y ** 2 - 7)])


def hessian_himmelblau(X):
	x, y = X
	return np.array([
		[12 * x ** 2 + 4 * y - 42, 4 * x + 4 * y],
		[4 * x + 4 * y, 4 * x + 12 * y ** 2 - 26]
	])
