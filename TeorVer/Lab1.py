import numpy as np
import matplotlib.pyplot as plt

# Заданные параметры
a1 = 0.2
a2 = -2
a3 = -1

# Нахождение константы c
def findC():
    return 1 / (((a3**2)/2 +a1*a3)-((a2**2)/2 +a1*a2))

# Плотность вероятности f(x)
def probFunc(x, c):
    if a2 <= x <= a3:
        return c * (x + a1)
    return 0.0

# Функция распределения F(x)
def disFunc(x, c):
    if x < a2:
        return 0.0
    if x > a3:
        return 1.0
    return c * ((x**2)/2 + a1*x) - c * ((a2**2)/2 + a1*a2)

# Математическое ожидание
def expectation(c):
    return c * (((a3 ** 3)/3 +(a1 * (a3 ** 2))/2) - ((a2 ** 3)/3 +(a1 * (a2 ** 2))/2))

# Дисперсия
def variance(c, mean):
    return c * (((a3 ** 4)/4 +(a1 * (a3 ** 3))/3) - ((a2 ** 4)/4 +(a1 * (a2 ** 3))/3)) - mean**2

# Медиана
def median(c):
    target = 0.5
    eps = 0.001
    for x in np.arange(a2, a3, 0.001):
        F_x = c * (0.5 * ((x + a1)**2 - (a2 + a1) ** 2))  
        if abs(F_x - target) < eps:  
            return x

# Мода
def mode(c):
    max = -1
    max_x = None
    for x in np.arange(a2, a3, 0.001):
        f_x = probFunc(x,c)
        if  f_x > max:
            max = f_x
            max_x = x
    return max_x



# Построение графиков
def plot_graphs(c, mean, median, mode):
    x_vals = np.linspace(a2 - 1, a3 + 1, 1000)
    pdf_vals = [probFunc(x, c) for x in x_vals]
    cdf_vals = [disFunc(x, c) for x in x_vals]


    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 5))

    # График плотности вероятности f(x)
    axes[0].plot(x_vals, pdf_vals, label="Плотность вероятности f(x)", color='r')
    axes[0].axvline(x=mean, color='green', linestyle='--', label=f"Математическое ожидание ({mean:.2f})")
    axes[0].axvline(x=median, color='orange', linestyle='--', label=f"Медиана ({median:.2f})")
    axes[0].axvline(x=mode, color='purple', linestyle='--', label=f"Мода ({mode:.2f})")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("Плотность вероятности")
    axes[0].set_title("График плотности вероятности f(x)")
    axes[0].legend()
    axes[0].grid(True)

    # График функции распределения F(x)
    axes[1].plot(x_vals, cdf_vals, label="Функция распределения F(x)", color='b')
    axes[1].axvline(x=median, color='orange', linestyle='--', label=f"Медиана ({median:.2f})")
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("F(x)")
    axes[1].set_title("График функции распределения F(x)")
    axes[1].legend()
    axes[1].grid(True)


    plt.tight_layout()
    plt.show()


def main():
    c = findC()
    mean = expectation(c)
    var = variance(c, mean)
    stddev = np.sqrt(var)
    med = median(c)
    mod = mode(c)

    # Вывод результатов
    print(f"Константа c: {c:.4f}")
    print(f"Математическое ожидание: {mean:.4f}")
    print(f"Дисперсия: {var:.4f}")
    print(f"Среднеквадратичное отклонение: {stddev:.4f}")
    print(f"Медиана: {med}")
    print(f"Мода: {mod}")

    # Построение графиков
    plot_graphs(c, mean, med, mod)

if __name__ == "__main__":
    main()
