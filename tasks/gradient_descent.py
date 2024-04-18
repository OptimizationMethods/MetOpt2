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
def golden_section_search(f: Callable[[Real], Real], eps: Real = 10 ** -12) -> tuple[Real, Real]:
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

        f_new1 = f(np.array([x1, 0]))
        f_new2 = f(np.array([x2, 0]))

        if f_new1 < f_new2:
            right = x2
        else:
            left = x1

    return (left + right) / 2, iterations

import numpy as np


def golden_section_search_mult(f, x, y, grad, epsilon=1e-12):
    phi = (1 + np.sqrt(5)) / 2
    resphi = 2 - phi

    a = -1
    b = 1

    while abs(b - a) > epsilon:
        x1 = a + resphi * (b - a)
        x2 = b - resphi * (b - a)

        f1 = f(np.array([x - x1 * grad[0], y - x1 * grad[1]]))
        f2 = f(np.array([x - x2 * grad[0], y - x2 * grad[1]]))

        if f1 < f2:
            b = x2
        else:
            a = x1

    return (a + b) / 2


@enhanced_time_profiler
def ternary_search(f: Callable[[Real], Real], eps: Real = 10 ** -12) -> tuple[Real, Real]:
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

        f1 = f(np.array([m1, 0]))
        f2 = f(np.array([m2, 0]))

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
        eps: Real = 10 ** -12,
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


def gradient_descent_mult(f, grad, X, selection_method, epsilon=1e-8,
                          num_iterations=1000):
    prev_points = X

    start_time = time.time()
    i = 0
    points = [prev_points]
    cur_points = prev_points
    for _ in range(num_iterations):
        gradient = grad(prev_points)

        learning_rate_val = selection_method(f, *prev_points, gradient, epsilon)

        i += 1

        cur_points = [0] * len(prev_points)
        for j in range(len(prev_points)):
            cur_points[j] = prev_points[j] - learning_rate_val * gradient[j]
        points.append(np.array(cur_points))

        if abs(f(cur_points) - f(prev_points)) < epsilon:
            break
        prev_points = cur_points

    end_time = time.time()
    execution_time = end_time - start_time

    return cur_points[0], cur_points[1], i, execution_time, np.array(points)
