import numpy as np
from functions import himmelblau, grad_himmelblau


def bfgs(f, grad_f, x0, tol=1e-5, max_iter=1000):
    x = x0
    n = len(x0)
    H = np.eye(n)
    count = 0

    for i in range(max_iter):
        grad = grad_f(x)
        if np.linalg.norm(grad) < tol:
            break

        count += 1

        p = -H @ grad  # Направление спуска
        # Линейный поиск с условием Вольфе (можно использовать scipy.optimize.line_search)
        alpha = 0.1

        x_new = x + alpha * p
        s = x_new - x
        y = grad_f(x_new) - grad

        rho = 1.0 / (y @ s)
        I = np.eye(n)
        H = (I - rho * np.outer(s, y)) @ H @ (I - rho * np.outer(y, s)) + rho * np.outer(s, s)

        x = x_new

    return x, count


x0 = np.array([0, 0])
result = bfgs(himmelblau, grad_himmelblau, x0)
print("Функция Химмельблау:")
print("Результат оптимизации самописного BFGS:", result[0], f"Количество итераций={result[1]}", end="\n\n")
