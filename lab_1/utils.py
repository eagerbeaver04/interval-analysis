import numpy as np
from interval_module import Interval


def get_interval_matrix(eps: float):
    return np.array([[Interval(1.05 - eps, 1.05 + eps), Interval(0.95 - eps, 0.95 + eps)],
                     [Interval(1 - eps, 1 + eps), Interval(1 - eps, 1 + eps)]])


def print_matrix_for_latex(matrix, index=0):
    print("\\begin{equation}\n\\text A_%d = \\begin{pmatrix}" % index)
    for items in matrix:
        print("&".join([item.__repr__() for item in items]) + "\\\\\n")
    print("\end{pmatrix}\n\end{equation}")


# Finding the maximum average value in a matrix
def find_max_middle(matrix):
    max_mid = -float("inf")
    for row in matrix:
        for interval in row:
            max_mid = max(max_mid, interval.mid())
    return max_mid


def determ(i, j, matrix):
    return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]


def optimize(left, right, delta) -> (float, float):
    counter = 0
    while right - left > delta:
        c = (right + left) / 2
        counter += 1
        matrix_tmp = get_interval_matrix(c)
        interval = determ(0, 0, matrix_tmp)

        if counter < 5 or counter == 34:
            print("-" * 20)
            print(f"$\\delta = {c}$")
            print(f"Number: {counter}:\n{print_matrix_for_latex(matrix_tmp, counter)}"
                  f"\nИтоговый интервал из определителя{interval}")

        if not 0 in interval:
            left = c
            print("-" * 20 + "\n" + "-" * 20)
            print(f"$\\delta = {c}$")
            print(f"Number: {counter}:\n{print_matrix_for_latex(matrix_tmp, counter)}"
                  f"\nИтоговый интервал из определителя{interval}")
        else:
            right = c

        print(f"Текущие границы [{left}, {right}]")
    return right, left, counter


def determinant_optimization_new(matrix=None, delta=1e-5):
    if matrix is None:
        matrix = get_interval_matrix(0)

    mid = find_max_middle(matrix)

    eps_curr = mid
    eps_left_bound = 0
    counter = 1

    eps_curr, eps_left_bound, amount = optimize(eps_left_bound, eps_curr, delta)

    print(f"Кол-во вызовов функции: {counter + 1}")
    return eps_curr


determinant_optimization_new(delta=1e-10)
