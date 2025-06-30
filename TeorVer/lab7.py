import numpy as np
import matplotlib.pyplot as plt

z = [0.6, 1.8, 1.9, 2.6, 4.0]
N = len(z)

def calculate_coefficients(M):
    A = [[0] * M for _ in range(M)]
    B = []
    for i in range(M):
        for j in range(M):
            A[i][j] = sum([k ** (i + j) for k in range(1, N + 1)])
        B.append(sum(z[j] * (j + 1) ** i for j in range(N)))

    return np.linalg.solve(A, B).tolist()

def calculate_estimates(coefficients):
    estimates = [0]*N
    for i in range(N):
        for j in range(len(coefficients)):
            estimates[i] += coefficients[j] * (i + 1) ** j
    return estimates

def calculate_std_estimate(estimates):
    N = len(estimates)
    squared_diff_sum = sum([(z[i] - estimates[i]) ** 2 for i in range(N)])
    std_estimate = (squared_diff_sum / (N - 1)) ** 0.5
    return std_estimate

def plot_data_and_polynomial(z, estimates, M):
    plt.plot(np.arange(1, N + 1), z, label='Исходные данные',color='blue')
    plt.scatter(np.arange(1, N + 1), z, color='blue')
    plt.plot(np.arange(1, N + 1), estimates, label=f'Аппроксимирующий полином (M={M})', color='red')
    plt.xlabel('Номер наблюдения')
    plt.ylabel('Значение')
    plt.title('Исходные данные и аппроксимирующий полином')
    plt.legend()
    plt.grid(True)
    plt.show()

def solve_exercise(M_values):
    for M in M_values:
        coefficients = calculate_coefficients(M)
        estimates = calculate_estimates(coefficients)
        std_estimate = calculate_std_estimate(estimates)
        print(f"M = {M}:")
        print("Коэффициенты:", [round(value, 2) for value in coefficients])
        print("Оценки:", [round(value, 2) for value in estimates])
        print("Оценка стандартного отклонения:", round(std_estimate, 3))
        plot_data_and_polynomial(z, estimates, M)

M_values = [1, 2, 3]
solve_exercise(M_values)
