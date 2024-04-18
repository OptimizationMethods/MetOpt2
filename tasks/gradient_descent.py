"""
Methods from Lab #1
"""


import time
import numpy as np

from typing import Callable, Optional
from numpy.typing import NDArray
from numbers import Real

from methods import enhanced_time_profiler


@enhanced_time_profiler
def golden_section_search(f: Callable[[Real], Real], eps: Real = 10**-12) -> tuple[Real, Real]:
	"""
	Golden section based search algorithm

	:param f: investigated function
	:param eps: precision
	:return: minimum of function
	"""
	phi = (1 + np.sqrt(5)) / 2
	phi2 = 2 - phi
	left = 0
	right = 1
	iterations = 0
	while abs(right - left) > eps:
		iterations += 1
		x1 = left + phi2 * (right - left)
		x2 = right - phi2 * (right - left)

		f_new1 = f(x1)
		f_new2 = f(x2)

		if f_new1 < f_new2:
			right = x2
		else:
			left = x1

	return (left + right) / 2, iterations


@enhanced_time_profiler
def ternary_search(f: Callable[[Real], Real], eps: Real = 10**-12) -> tuple[Real, Real]:
	"""
	Ternary search for function minimum

	:param f: investigated function
	:param eps: precision
	:return: minimum of function
	"""
	left = 0
	right = 1
	iterations = 0
	while abs(right - left) > eps:
		iterations += 1
		m1 = left + (right - left) / 3
		m2 = right - (right - left) / 3

		f1 = f(m1)
		f2 = f(m2)

		if f1 < f2:
			right = m2
		else:
			left = m1

	return (left + right) / 2, iterations


@enhanced_time_profiler
def gradient_descent(
		f: Callable,
		grad: Callable,
		X: NDArray,
		selection_method: Callable,
		eps: Real = 10**-12,
		learning_rate: Optional[Real] = None,
		max_iter: int = 1500
):
	"""
	Implements the gradient descent optimization algorithm.

	:param X: point
	:param f: Function that the algorithm is trying to minimize.
	:param grad: Function that computes the gradient of 'f'.
	:param selection_method: Method for selecting the step size.
	:param eps: A real number representing the precision of the solution. The algorithm stops when the absolute difference between the function values at two consecutive iterations is less than 'eps'.
	:param learning_rate: An optional real number representing the step size at each iteration. If not provided, the 'selection_method' is used to determine the step size.
	:param max_iter: Maximum number of iterations.
	:return:
	"""
	x0, y0 = X
	x_prev = x0
	y_prev = y0
	trajectory = [(x0, y0)]

	iter_count = 0

	start_time = time.time()
	x = x_prev
	y = y_prev
	for _ in range(max_iter):
		grad_f = grad(x_prev, y_prev)
		if learning_rate is not None:
			alpha = learning_rate
		else:
			f_alpha = lambda al: f(x_prev - al * grad_f[0], y_prev - al * grad_f[1])
			alpha = selection_method(f_alpha, eps)
		x = x_prev - alpha * grad_f[0]
		y = y_prev - alpha * grad_f[1]

		trajectory.append((x, y))

		iter_count += 1

		if abs(f(x, y) - f(x_prev, y_prev)) < eps:
			break
		x_prev = x
		y_prev = y

	end_time = time.time()
	exec_time = end_time - start_time

	return x, y, iter_count, exec_time, trajectory
