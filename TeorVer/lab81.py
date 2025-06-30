import numpy as np
import matplotlib.pyplot as plt

# Исходные данные
x = np.array([2, 3, 4, 5, 7], dtype=float)
y = np.array([7, 5, 5, 4, 2], dtype=float)
N = len(x)

#y = a_1 + a_2 * cos(x) + a_3 * cos(2 * x)
X_cos = np.vstack([np.ones_like(x), np.cos(x), np.cos(2*x)]).T
coeff_cos = np.linalg.inv(X_cos.T @ X_cos) @ (X_cos.T @ y)

x_fore = np.array([8, 9], dtype=float)
Xf_cos = np.vstack([np.ones_like(x_fore), np.cos(x_fore), np.cos(2*x_fore)]).T
yf_cos = Xf_cos @ coeff_cos

x_dense = np.linspace(x.min(), 10, 200)
X_dense_cos = np.vstack([np.ones_like(x_dense), np.cos(x_dense), np.cos(2*x_dense)]).T
y_cos_dense = X_dense_cos @ coeff_cos

plt.figure(figsize=(10, 6))
plt.scatter(x, y, label='Исходные данные', color='blue', s=100)
plt.plot(x_dense, y_cos_dense, 'r-', label='Косинусная модель', linewidth=2)
plt.scatter(x_fore, yf_cos, marker='x', label='Прогноз', color='orange', s=200, linewidth=3)
plt.title('Аппроксимация функцией: y = a_1 + a_2 * cos(x) + a_3 * cos(2 * x)', fontsize=14)
plt.xlabel('x', fontsize=12)
plt.ylabel('y', fontsize=12)
plt.legend(fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.show()

print("Коэффициенты модели:")
print(f"a₁ = {coeff_cos[0]:.4f}")
print(f"a₂ = {coeff_cos[1]:.4f}")
print(f"a₃ = {coeff_cos[2]:.4f}")

print("\nПрогнозные значения:")
for xi, yi in zip(x_fore, yf_cos):
    print(f"x = {xi}: y = {yi:.4f}")