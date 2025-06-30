import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.optimize import fsolve
import pandas as pd

class SpacecraftTemperatureAnalyzer:
    def __init__(self, N=13):
        self.N = N
        self.set_parameters()
        
    def set_parameters(self):
        """Установка параметров согласно варианту"""
        # Параметры нормальных распределений из таблицы
        self.x0_bar = 23  # Среднее для исправного состояния (из таблицы для N=13)
        self.sigma0 = 3.5  # СКО для исправного состояния
        self.x1_bar = 35   # Среднее для неисправного состояния
        self.sigma1 = 5.2  # СКО для неисправного состояния
        
        # Априорные вероятности
        self.P0 = 0.97  # Вероятность исправного состояния
        self.P1 = 0.03  # Вероятность неисправного состояния
        
        # Параметры ошибок
        self.alpha = self.N / 100
        self.beta = self.N / 200
        
        # Матрица потерь
        self.C00 = 0
        self.C11 = 0
        self.C01 = 20 * (self.N + 5) / self.N
        self.C10 = (self.N + 5) / self.N
        
    def pdf_normal(self, x, mu, sigma):
        """Функция плотности нормального распределения"""
        return (1/(sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu)/sigma)**2)
    
    def calculate_threshold_min_risk(self):
        """Расчет граничного значения по методу минимального среднего риска"""
        # Коэффициенты квадратного уравнения ax² + bx + c = 0
        a = (1/self.sigma0**2 - 1/self.sigma1**2)
        b = 2 * (self.x1_bar/self.sigma1**2 - self.x0_bar/self.sigma0**2)
        c = (self.x0_bar**2/self.sigma0**2 - self.x1_bar**2/self.sigma1**2 - 
             2 * np.log((self.sigma0 * (self.C01 - self.C11) * self.P1) / 
                        (self.sigma1 * (self.C10 - self.C00) * self.P0)))
        
        # Решение квадратного уравнения
        discriminant = b**2 - 4*a*c
        x1 = (-b + np.sqrt(discriminant)) / (2*a)
        x2 = (-b - np.sqrt(discriminant)) / (2*a)
        
        # Выбираем значение между x0_bar и x1_bar
        if self.x0_bar < x1 < self.x1_bar:
            return x1
        else:
            return x2
    
    def calculate_threshold_min_error(self):
        """Расчет граничного значения по методу минимальной вероятности ошибки"""
        # Коэффициенты квадратного уравнения ax² + bx + c = 0
        a = (1/self.sigma0**2 - 1/self.sigma1**2)
        b = 2 * (self.x1_bar/self.sigma1**2 - self.x0_bar/self.sigma0**2)
        c = (self.x0_bar**2/self.sigma0**2 - self.x1_bar**2/self.sigma1**2 - 
             2 * np.log((self.sigma0 * self.P1) / (self.sigma1 * self.P0)))
        
        # Решение квадратного уравнения
        discriminant = b**2 - 4*a*c
        x1 = (-b + np.sqrt(discriminant)) / (2*a)
        x2 = (-b - np.sqrt(discriminant)) / (2*a)
        
        # Выбираем значение между x0_bar и x1_bar
        if self.x0_bar < x1 < self.x1_bar:
            return x1
        else:
            return x2
    
    def calculate_probabilities(self, x_threshold):
        """Расчет вероятностей ошибок для заданного порога"""
        # Вероятность ложной тревоги (ошибка 1-го рода)
        alpha = 1 - norm.cdf(x_threshold, loc=self.x0_bar, scale=self.sigma0)
        P_H01 = self.P0 * alpha
        
        # Вероятность пропуска дефекта (ошибка 2-го рода)
        beta = norm.cdf(x_threshold, loc=self.x1_bar, scale=self.sigma1)
        P_H10 = self.P1 * beta
        
        return alpha, P_H01, beta, P_H10
    
    def calculate_risk(self, P_H01, P_H10, method='min_risk'):
        """Расчет среднего риска"""
        if method == 'min_risk':
            return self.C10 * P_H01 + self.C01 * P_H10
        else:
            return P_H01 + P_H10
    
    def fill_initial_table(self):
        """Заполнение таблицы 2 (параметры состояния КА)"""
        data = {
            'Состояние': ['S0', 'S1'],
            'Априорная вероятность': [self.P0, self.P1],
            'Принятое решение': [['H00', 'H10'], ['H01', 'H11']],
            'Оценка решения': [['правильное', 'ошибка 1-го рода'], 
                              ['ошибка 2-го рода', 'правильное']],
            'Вероятность решения': [[f'1 - α = {1-self.alpha:.3f}', f'α = {self.alpha:.3f}'], 
                                  [f'β = {self.beta:.3f}', f'1 - β = {1-self.beta:.3f}']],
            'Стоимость потерь': [[self.C00, self.C10], [self.C01, self.C11]],
            'Средний риск': [
                [f'{self.P0 * self.C00 * (1-self.alpha):.4f}', 
                 f'{self.P0 * self.C10 * self.alpha:.4f}'],
                [f'{self.P1 * self.C01 * self.beta:.4f}', 
                 f'{self.P1 * self.C11 * (1-self.beta):.4f}']
            ]
        }
        return pd.DataFrame(data)
    
    def analyze(self):
        """Полный анализ и вывод результатов"""
        # 1. Метод минимального среднего риска
        x_thresh_risk = self.calculate_threshold_min_risk()
        alpha_risk, P_H01_risk, beta_risk, P_H10_risk = self.calculate_probabilities(x_thresh_risk)
        risk_risk = self.calculate_risk(P_H01_risk, P_H10_risk, 'min_risk')
        
        # 2. Метод минимальной вероятности ошибки
        x_thresh_error = self.calculate_threshold_min_error()
        alpha_error, P_H01_error, beta_error, P_H10_error = self.calculate_probabilities(x_thresh_error)
        risk_error = self.calculate_risk(P_H01_error, P_H10_error, 'min_error')
        
        # Создаем таблицу результатов
        results = pd.DataFrame({
            'Метод': [
                'Метод минимального среднего риска',
                'Метод минимальной вероятности ошибочного решения'
            ],
            'Граничное значение': [x_thresh_risk, x_thresh_error],
            'Вероятность ложной тревоги': [P_H01_risk, P_H01_error],
            'Вероятность пропуска дефекта': [P_H10_risk, P_H10_error],
            'Средний риск': [risk_risk, risk_error]
        })
        
        return results
    
    def plot_distributions(self):
        """Визуализация распределений и пороговых значений"""
        x_values = np.linspace(self.x0_bar - 4*self.sigma0, self.x1_bar + 4*self.sigma1, 500)
        pdf0 = self.pdf_normal(x_values, self.x0_bar, self.sigma0)
        pdf1 = self.pdf_normal(x_values, self.x1_bar, self.sigma1)
        
        # Вычисляем пороговые значения
        x_thresh_risk = self.calculate_threshold_min_risk()
        x_thresh_error = self.calculate_threshold_min_error()
        
        plt.figure(figsize=(12, 6))
        plt.plot(x_values, pdf0, label=f'Исправное состояние (μ={self.x0_bar}, σ={self.sigma0})')
        plt.plot(x_values, pdf1, label=f'Неисправное состояние (μ={self.x1_bar}, σ={self.sigma1})')
        
        # Пороговые линии
        plt.axvline(x_thresh_risk, color='r', linestyle='--', 
                   label=f'Порог по мин. риску: {x_thresh_risk:.2f}')
        plt.axvline(x_thresh_error, color='g', linestyle='--', 
                   label=f'Порог по мин. ошибке: {x_thresh_error:.2f}')
        
        plt.title('Распределения температуры при разных состояниях КА')
        plt.xlabel('Температура')
        plt.ylabel('Плотность вероятности')
        plt.legend()
        plt.grid(True)
        plt.show()

# Основная программа
if __name__ == "__main__":
    analyzer = SpacecraftTemperatureAnalyzer(N=13)
    
    # 1. Вывод таблицы параметров состояния
    print("Таблица 2. Параметры состояния КА:")
    initial_table = analyzer.fill_initial_table()
    print(initial_table.to_string(index=False))
    
    # 2. Расчет вероятности ошибочного решения
    P_error = analyzer.P0 * analyzer.alpha + analyzer.P1 * analyzer.beta
    print(f"\nВероятность ошибочного решения: {P_error:.4f}")
    
    # 3. Расчет среднего риска при известных α и β
    avg_risk = (analyzer.P0 * analyzer.C00 * (1 - analyzer.alpha) + 
               analyzer.P0 * analyzer.C10 * analyzer.alpha + 
               analyzer.P1 * analyzer.C01 * analyzer.beta + 
               analyzer.P1 * analyzer.C11 * (1 - analyzer.beta))
    print(f"Средний риск при известных α и β: {avg_risk:.4f} усл. ед.")
    
    # 4. Полный анализ и вывод результатов
    print("\nТаблица 3. Определение граничного значения температуры и среднего риска:")
    results = analyzer.analyze()
    print(results.to_string(index=False))
    
    # 5. Визуализация распределений
    analyzer.plot_distributions()