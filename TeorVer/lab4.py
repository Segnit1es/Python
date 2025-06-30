import numpy as np
import scipy.stats as stats

N = 13
alpha = N / 100
beta = N / 200
P0 = 0.97
P1 = 0.03

x0_bar = 23
sigma0 = 3.5
x1_bar = 35
sigma1 = 5.2

C00 = 0
C11 = 0
C01 = 20 * (N + 5) / N
C10 = (N + 5) / N

print("\nПараметры состояния КА при известных значениях параметров alpha и beta:")
print(f"alpha = {alpha:.3f}, beta = {beta:.3f}")
print("\nВероятности решений:")
print(f"P(H00) = {1 - alpha:.3f}")
print(f"P(H10) = {alpha:.3f}")
print(f"P(H01) = {beta:.3f}")
print(f"P(H11) = {1 - beta:.3f}")
print("\nПотери:")
print(f"C00 = {C00}, C10 = {C10:.3f}, C01 = {C01:.3f}, C11 = {C11}")

R00 = P0 * C00 * (1 - alpha)
R10 = P0 * C10 * alpha
R01 = P1 * C01 * beta
R11 = P1 * C11 * (1 - beta)
total_risk = R00 + R10 + R01 + R11
print("\nСредний риск по компонентам:")
print(f"R00 = {R00:.4f}, R10 = {R10:.4f}, R01 = {R01:.4f}, R11 = {R11:.4f}")
print(f"Общий средний риск: {total_risk:.4f}")

def bayes_risk_threshold(x0, sigma0, x1, sigma1, P0, P1, C10, C01):
    k = (P1 * (C01 - C11)) / (P0 * (C10 - C00))
    a = 1/(2*sigma1**2) - 1/(2*sigma0**2)
    b = x0/sigma0**2 - x1/sigma1**2
    c = (x1**2)/(2*sigma1**2) - (x0**2)/(2*sigma0**2) - np.log((sigma0/sigma1)*k)
    
    discriminant = b**2 - 4*a*c
    if discriminant < 0:
        return None
    x1 = (-b + np.sqrt(discriminant)) / (2*a)
    x2 = (-b - np.sqrt(discriminant)) / (2*a)
    
    if x0 < x1 < x1_bar:
        return x1
    elif x0 < x2 < x1_bar:
        return x2
    else:
        return (x0 + x1_bar)/2

def min_error_threshold(x0, sigma0, x1, sigma1, P0, P1):
    k = P1/P0
    a = 1/(2*sigma1**2) - 1/(2*sigma0**2)
    b = x0/sigma0**2 - x1/sigma1**2
    c = (x1**2)/(2*sigma1**2) - (x0**2)/(2*sigma0**2) - np.log((sigma0/sigma1)*k)
    
    discriminant = b**2 - 4*a*c
    if discriminant < 0:
        return None
    x1 = (-b + np.sqrt(discriminant)) / (2*a)
    x2 = (-b - np.sqrt(discriminant)) / (2*a)
    
    if x0 < x1 < x1_bar:
        return x1
    elif x0 < x2 < x1_bar:
        return x2
    else:
        return (x0 + x1_bar)/2

x_risk = bayes_risk_threshold(x0_bar, sigma0, x1_bar, sigma1, P0, P1, C10, C01)
if x_risk is not None:
    alpha_risk = 1 - stats.norm.cdf(x_risk, loc=x0_bar, scale=sigma0)
    beta_risk = stats.norm.cdf(x_risk, loc=x1_bar, scale=sigma1)
    R_risk = P0 * C10 * alpha_risk + P1 * C01 * beta_risk
else:
    alpha_risk, beta_risk, R_risk = None, None, None

x_error = min_error_threshold(x0_bar, sigma0, x1_bar, sigma1, P0, P1)
if x_error is not None:
    alpha_error = 1 - stats.norm.cdf(x_error, loc=x0_bar, scale=sigma0)
    beta_error = stats.norm.cdf(x_error, loc=x1_bar, scale=sigma1)
    R_error = P0 * C10 * alpha_error + P1 * C01 * beta_error
else:
    alpha_error, beta_error, R_error = None, None, None

print("\nОпределение граничного значения температуры xгр и среднего риска R:")
if x_risk is not None:
    print("Метод: Минимального среднего риска")
    print(f"  Граница температуры: {x_risk:.2f}")
    print(f"  Вероятность ложной тревоги (α): {alpha_risk:.4f}")
    print(f"  Вероятность пропуска дефекта (β): {beta_risk:.4f}")
    print(f"  Средний риск: {R_risk:.4f}")
else:
    print("Метод минимального среднего риска: не удалось найти действительное граничное значение")

if x_error is not None:
    print("\nМетод: Минимальной вероятности ошибки")
    print(f"  Граница температуры: {x_error:.2f}")
    print(f"  Вероятность ложной тревоги (α): {alpha_error:.4f}")
    print(f"  Вероятность пропуска дефекта (β): {beta_error:.4f}")
    print(f"  Средний риск: {R_error:.4f}")
else:
    print("\nМетод минимальной вероятности ошибки: не удалось найти действительное граничное значение")