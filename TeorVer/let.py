import numpy as np
import matplotlib.pyplot as plt

data = [5, 7, 6, 8, 10, 9, 11]
n = 3  

moving_avg = []
for i in range(len(data) - n + 1):
    window = data[i:i + n]
    window_avg = sum(window) / n
    moving_avg.append(window_avg)

x_original = np.arange(len(data))
x_moving_avg = np.arange(n - 1, len(data))  

plt.figure(figsize=(10, 5))
plt.plot(x_original, data, 'bo-', label='Исходные данные')
plt.plot(x_moving_avg, moving_avg, 'ro-', label=f'Скользящее среднее (n={n})')

plt.title('Исходные данные и скользящее среднее')
plt.xlabel('Временные точки')
plt.ylabel('Значения')
plt.legend()
plt.grid(True)
plt.xticks(x_original)
plt.show()