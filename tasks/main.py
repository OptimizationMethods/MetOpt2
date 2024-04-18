from functions import *

from methods import *

from gradient_descent import *


def compare_methods_1(X0: NDArray) -> None:
	"""
	Task 1 - сравните свою реализацию метода Ньютона с методом
	Newton-CG из библиотеки scipy.optimize (по точности и скорости)
	:param X0: start point
	"""
	newton_x, newton_iters = newton_method(f=rosenbrock, grad=grad_rosenbrock, hess=hessian_rosenbrock, x0=X0)
	scipy_x, scipy_iters = scipy_methods(method_type="Newton-CG", point=X0, function=rosenbrock,
	                                     grad_func=grad_rosenbrock, hessian=hessian_rosenbrock)


compare_methods_1(np.array([0, 0]))


def compare_methods_2(X0: NDArray) -> None:
	golden_x, golden_iter = golden_section_search(f=himmelblau)
	tern_x, tern_iter = ternary_search(f=himmelblau)
	grad_x, grad_y, grad_iter, grad_time, grad_traj = gradient_descent(f=himmelblau, grad=grad_himmelblau, X=X0,
	                                                                   selection_method=golden_section_search)
	newton_x, newton_iter = newton_one_dim_search(f=himmelblau, grad_f=grad_himmelblau, hessian_f=hessian_himmelblau, X0=X0)
	newton_x, newton_iter = scipy_methods(method_type="Newton-CG", point=X0, function=himmelblau, grad_func=grad_himmelblau, hessian=hessian_himmelblau)
	newton_x_BFGS, newton_iter_BFGS = scipy_methods(method_type="BFGS", point=X0, function=himmelblau, grad_func=grad_himmelblau, hessian=hessian_himmelblau)

