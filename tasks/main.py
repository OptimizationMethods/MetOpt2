import pandas as pd

from functions import *

from methods import *

from gradient_descent import *

from typing import Any


def compare_methods_1(X0: NDArray) -> dict[str, Any]:
    """
    Task 1 - сравните свою реализацию метода Ньютона с методом
    Newton-CG из библиотеки scipy.optimize (по точности и скорости)
    :param X0: start point
    """
    newton_x, newton_iters = newton_method(f=rosenbrock, grad=grad_rosenbrock, hess=hessian_rosenbrock, x0=X0)
    scipy_x, scipy_iters = scipy_methods(method_type="Newton-CG", point=X0, function=rosenbrock,
                                         grad_func=grad_rosenbrock, hessian=hessian_rosenbrock)
    return {"newton_x": newton_x, "newton_iters": newton_iters, "scipy_x": scipy_x, "scipy_iters": scipy_iters}


def compare_methods_2(X0: NDArray) -> dict[str, Real | Any]:
    """
    Task 2 - сравните эффективность методов нулевого порядка
    и градиентного спуска из лаб 1., метода Ньютона, квазиньютоновских методов.
    :param X0: start point
    """
    golden_x, golden_iter = golden_section_search(f=himmelblau)
    tern_x, tern_iter = ternary_search(f=himmelblau)
    grad_x, grad_y, grad_iter, grad_time, grad_traj = gradient_descent_mult(f=himmelblau, grad=grad_himmelblau, X=X0,
                                                                       selection_method=golden_section_search_mult)
    newton_x_our, newton_iter_our = newton_one_dim_search(f=himmelblau, grad_f=grad_himmelblau, hessian_f=hessian_himmelblau,
                                                  X0=X0)
    newton_x, newton_iter = scipy_methods(method_type="Newton-CG", point=X0, function=himmelblau,
                                          grad_func=grad_himmelblau, hessian=hessian_himmelblau)
    newton_x_BFGS, newton_iter_BFGS = scipy_methods(method_type="BFGS", point=X0, function=himmelblau,
                                                    grad_func=grad_himmelblau, hessian=hessian_himmelblau)
    return {"golden_x": golden_x, "golden_iter": golden_iter, "tern_x": tern_x, "tern_iter": tern_iter,
            "grad_x": grad_x, "grad_iter": grad_iter, "newton_x_our": newton_x_our, "newton_iter_our": newton_iter_our,
            "newton_x": newton_x, "newton_iter": newton_iter,
            "newton_x_BFGS": newton_x_BFGS, "newton_iter_BFGS": newton_iter_BFGS}


def compare_methods_3(X0: NDArray) -> dict[str, Any]:
    """
    Task 3 - сравните эффективность методов нулевого порядка с квазиньютоновскими методами,
    если в последних производная вычисляется разностным методом
    """
    grad_x, grad_y, grad_iter, grad_time, grad_traj = gradient_descent_mult(f=himmelblau, grad=grad_himmelblau, X=X0,
                                                                       selection_method=golden_section_search_mult)

    newton_x_BFGS, newton_iter_BFGS = scipy_methods(method_type="BFGS", point=X0, function=himmelblau,
                                                    jac='2-point',
                                                    hessian=hessian_himmelblau)
    return {"grad_x": grad_x, "grad_iter": grad_iter,
            "newton_x_BFGS": newton_x_BFGS, "newton_iter_BFGS": newton_iter_BFGS}


def display_results():
    X0 = np.array([0, 0])
    results_1 = compare_methods_1(X0)
    results_2 = compare_methods_2(X0)
    results_3 = compare_methods_3(X0)

    data1 = {
        "1": ["Our Newton", "Scipy Newton-CG"],
        "x": [results_1['newton_x'], results_1['scipy_x']],
        "iters": [results_1['newton_iters'], results_1['scipy_iters']]
    }


    data2 = {
        "Method": ["Golden Search", "Ternary Search", "Gradient Descent", "Our Newton", "Scipy Newton-CG", "Scipy BFGS"],
        "x": [results_2['golden_x'], results_2['tern_x'], results_2['grad_x'], results_2['newton_x_our'], results_2['newton_x'], results_2['newton_x_BFGS']],
        "iter": [results_2['golden_iter'], results_2['tern_iter'], results_2['grad_iter'], results_2['newton_iter_our'], results_2['newton_iter'], results_2['newton_iter_BFGS']]
    }

    data3 = {
        "Method": ["Gradient Descent", "Scipy BFGS"],
        "x": [results_3['grad_x'], results_3['newton_x_BFGS']],
        "grad_iter": [results_3['grad_iter'], results_3['newton_iter_BFGS']]
    }

    df1 = pd.DataFrame(data1)
    df2 = pd.DataFrame(data2)
    df3 = pd.DataFrame(data3)
    return df1, df2, df3
