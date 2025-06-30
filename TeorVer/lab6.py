import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

sensor1 = np.array([9.7, 9.5, 10.6, 10.5, 9.1, 9.4, 10.6, 10.4, 9.7, 9.7])
sensor2 = np.array([10.6, 9.9, 9.4, 9.1, 9.6, 11.4, 9.4, 9.1, 9.8, 10.5])
sensor3 = np.array([10.2, 10.5, 10.0, 10.0, 10.1, 10.3, 10.0, 9.5, 9.7, 11.0])
sensor4 = np.array([9.8, 10.5, 9.8, 11.2, 10.4, 10.2, 9.6, 10.2, 10.4, 10.1])

all_data = [sensor1, sensor2, sensor3, sensor4]
y_all = np.concatenate(all_data)
overall_mean = np.mean(y_all)
means = [np.mean(group) for group in all_data]

SS_total = np.sum((y_all - overall_mean)**2)
SS_between = sum(len(group) * (group_mean - overall_mean)**2 
             for group, group_mean in zip(all_data, means))
SS_within = sum(((group - group_mean)**2).sum() 
               for group, group_mean in zip(all_data, means))

assert np.isclose(SS_total, SS_between + SS_within), "Ошибка в расчетах сумм квадратов"

N = len(y_all) 
m = len(all_data)  

D_total = SS_total / (N - 1)
D_between = SS_between / (m - 1)
D_within = SS_within / (N - m)

F_calc = D_between / D_within
df_between = m - 1
df_within = N - m
F_crit = stats.f.ppf(1 - 0.05, df_between, df_within)

conclusion = "F < F_crit, H0 не отвергаем" if F_calc < F_crit else "F ≥ F_crit, H0 отвергается"

f_stat, p_val = stats.f_oneway(sensor1, sensor2, sensor3, sensor4)

results = pd.DataFrame({
    "Параметр": [
        "Общая дисперсия (D_total)",
        "Межгрупповая дисперсия (Dмг)", 
        "Внутригрупповая дисперсия (Dвг)",
        "F-статистика (расч.)",
        "F-статистика (scipy)",
        "Критическое F (α=0.05)",
        "p-value",
        "Вывод"
    ],
    "Значение": [
        round(D_total, 4),
        round(D_between, 4),
        round(D_within, 4),
        round(F_calc, 4),
        round(f_stat, 4),
        round(F_crit, 4),
        round(p_val, 4),
        conclusion
    ]
})

print("\nРезультаты дисперсионного анализа:")
print(results)