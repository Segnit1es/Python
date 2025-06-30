import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Данные
X = np.array([5, 10, 15, 20, 25, 30, 35, 40, 45, 50,
              55, 60, 65, 70, 75, 80, 85, 90, 95, 100])
Y = np.array([50, 65, 70, 75, 80, 85, 90, 92, 95, 98,
              100, 100, 100, 100, 100, 100, 100, 100, 100, 100])

# Описательная статистика
def descriptive_stats(data):
    return {
        'mean': np.mean(data),
        'median': np.median(data),
        'variance': np.var(data, ddof=1),
        'std_dev': np.std(data, ddof=1)
    }

desc_X = descriptive_stats(X)
desc_Y = descriptive_stats(Y)

print("Описательная статистика:")
print("X (время подготовки):", desc_X)
print("Y (результат экзамена):", desc_Y)