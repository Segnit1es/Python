import numpy as np
import matplotlib.pyplot as plt

# Данные
x = np.array([2, 3, 4, 5, 7], dtype=float)
y = np.array([7, 5, 5, 4, 2], dtype=float)
N = len(x)

# Линейная модель
X_lin = np.vstack([np.ones_like(x), x]).T
coeff_lin = np.linalg.inv(X_lin.T @ X_lin) @ (X_lin.T @ y)
x_fore = np.array([8, 9], dtype=float)
Xf_lin = np.vstack([np.ones_like(x_fore), x_fore]).T
yf_lin = Xf_lin @ coeff_lin


x_dense = np.linspace(x.min(), 10, 200)
X_dense_lin = np.vstack([np.ones_like(x_dense), x_dense]).T
y_dense_lin = X_dense_lin @ coeff_lin


plt.figure(figsize=(6, 4))
plt.scatter(x, y, label='Исходные данные')
plt.plot(x_dense, y_dense_lin, '--', label='Линейная аппроксимация')
plt.scatter(x_fore, yf_lin, marker='x', label='Прогноз', color='orange')
plt.title('Линейная модель')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.show()

# Квадратичная модель
X_quad = np.vstack([np.ones_like(x), x, x**2]).T
coeff_quad = np.linalg.inv(X_quad.T @ X_quad) @ (X_quad.T @ y)
Xf_quad = np.vstack([np.ones_like(x_fore), x_fore, x_fore**2]).T
yf_quad = Xf_quad @ coeff_quad


x_dense = np.linspace(x.min(), 10, 200)
X_dense_quad = np.vstack([np.ones_like(x_dense), x_dense, x_dense**2]).T
y_quad_dense = X_dense_quad @ coeff_quad

plt.figure(figsize=(6, 4))
plt.scatter(x, y, label='Исходные данные')
plt.plot(x_dense, y_quad_dense, '-', label='Квадратичная аппрокс.')
plt.scatter(x_fore, yf_quad, marker='o', label='Прогноз', color='orange')
plt.title('Квадратичная модель')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.show()
