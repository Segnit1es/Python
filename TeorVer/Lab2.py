import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Параметр c
c = 13

def Mean(z):
    return np.mean(z)

def HalfSum(z):
    return (np.min(z) + np.max(z)) / 2

def Median(z):
    return np.median(z)

def TrimmeredMean(z):
    k = 1 if len(z) < 10 else (2 if len(z) < 15 else 3)
    return np.mean(sorted(z)[k:-k])

algorithms = {
    "Выборочное среднее": Mean,
    "Полусумма макс и мин": HalfSum,
    "Медиана": Median,
    "Среднее с отбросом": TrimmeredMean,
}

distributions = [
    ("z=c+N(0,c/100)", lambda n: c + np.random.normal(0, c/100, n)),
    ("z=c+N(0,c/20)", lambda n: c + np.random.normal(0, c/20, n)),
    ("z=c+U(-c/100,c/100)", lambda n: c + np.random.uniform(-c/100, c/100, n)),
    ("z=c+U(-c/20,c/20)", lambda n: c + np.random.uniform(-c/20, c/20, n)),
]

sampleSize = [15, 30, 100, 1000]
experiments = 50

for distName, distFunc in distributions:
    fig, axes = plt.subplots(1, 4, figsize=(16, 4), sharey=True)
    fig.suptitle(f"Model: {distName}")
    
    for ax, (alg_name, alg_func) in zip(axes, algorithms.items()):
        for n in sampleSize:
            estimates = [alg_func(distFunc(n)) for _ in range(experiments)]
            ax.scatter([n] * experiments, estimates, label=f"n={n}", s=5)
        
        ax.axhline(y=c, color='blue', linestyle='--')
        ax.set_xscale('log')
        ax.set_xlabel("Sampling size")
        ax.set_title(alg_name)
        ax.legend()
    
    plt.show()

results = []
for distName, distFunc in distributions:
    for n in sampleSize:
        errorMean = []
        for alg_name, alg_func in algorithms.items():
            errorList = []
            for _ in range(experiments):
                z = distFunc(n)
                x_hat = alg_func(z)
                error = abs((c - x_hat) / c)
                errorList.append(error)
            errorMean.append(np.mean(errorList))
        results.append([distName, n] + errorMean)

columns = ["Модель шума", "Объем выборки", "σx1", "σx2", "σx3", "σx3"]
df = pd.DataFrame(results, columns=columns)

print(df)
