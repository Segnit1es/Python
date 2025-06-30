import numpy as np
import matplotlib.pyplot as plt
import json
import pandas as pd
from scipy.stats import norm, chi2
from collections import Counter

N = 13
a, b = -N / 10, N / 2
M_n, sigma_n = N, N / 3
l1, l2 = 100, 1000
file_path = "random_samples.json"
alpha = 0.07

def GenerateNormalBoxMuller(size):
    adjusted_size = size + (size % 2)
    u1, u2 = np.random.rand(adjusted_size // 2), np.random.rand(adjusted_size // 2)
    z1 = np.sqrt(-2 * np.log(u1)) * np.cos(2 * np.pi * u2)
    z2 = np.sqrt(-2 * np.log(u1)) * np.sin(2 * np.pi * u2)
    return np.concatenate([z1, z2])[:size]

try:
    with open(file_path, "r") as f:
        data = json.load(f)
    x1, x2 = np.array(data["x1"]), np.array(data["x2"])
    n1, n2 = np.array(data["n1"]), np.array(data["n2"])
except (FileNotFoundError, KeyError, json.JSONDecodeError):
    np.random.seed(42)
    x1 = a + (b - a) * np.random.rand(l1)
    x2 = a + (b - a) * np.random.rand(l2)
    n1 = M_n + sigma_n * GenerateNormalBoxMuller(l1)
    n2 = M_n + sigma_n * GenerateNormalBoxMuller(l2)
    with open(file_path, "w") as f:
        json.dump({
            "x1": x1.tolist(), "x2": x2.tolist(),
            "n1": n1.tolist(), "n2": n2.tolist(),
            "parameters": {"N": N, "a": a, "b": b, "M_n": M_n, "sigma_n": sigma_n}
        }, f, indent=4)

def ComputeStatistics(sample):
    values = np.round(sample, 2)
    mode = Counter(values).most_common(1)[0][0]
    return {
        "mean": np.mean(sample),
        "median": np.median(sample),
        "mode": float(mode),
        "variance": np.var(sample, ddof=1),
        "std_dev": np.std(sample, ddof=1)
    }

def PrintStatistics(name, stats):
    print(f"\nСтатистики для {name}:")
    for key, value in stats.items():
        print(f"{key:10}: {value:.4f}")

def ComputeHistogram(sample, q):
    R = np.ptp(sample)
    delta = R / q
    bins = np.linspace(np.min(sample), np.max(sample), q + 1)
    hist, _ = np.histogram(sample, bins=bins)
    print(f"\nТаблица ({q} интервалов):")
    print("№  [x_j ; x_{j+1})     n_j")
    for i in range(q):
        print(f"{i+1:<3} [{bins[i]:.2f} ; {bins[i+1]:.2f})  {hist[i]}")
    print(f"Сумма: {sum(hist)}")
    return hist, bins

def ChiSquareTable(sample, bins_count=7, alpha=0.07):
    N = len(sample)
    mean = np.mean(sample)
    std = np.std(sample, ddof=1)
    hist, bin_edges = np.histogram(sample, bins=bins_count)
    intervals = list(zip(bin_edges[:-1], bin_edges[1:]))

    results, chi2_stat = [], 0
    for j, (xj, xj1) in enumerate(intervals):
        Fj, Fj1 = norm.cdf(xj, loc=mean, scale=std), norm.cdf(xj1, loc=mean, scale=std)
        pj, npj, nj = Fj1 - Fj, N * (Fj1 - Fj), hist[j]
        diff_sq = (nj - npj)**2
        chi_contrib = diff_sq / npj if npj > 0 else 0
        chi2_stat += chi_contrib
        results.append({
            "№ инт.": j + 1, "x_j": round(xj, 4), "x_j+1": round(xj1, 4),
            "n_j": nj, "F(x_j)": round(Fj, 4), "F(x_j+1)": round(Fj1, 4),
            "p_j": round(pj, 4), "N*p_j": round(npj, 2),
            "(n_j - Np_j)^2": round(diff_sq, 2),
            "(n_j - Np_j)^2 / Np_j": round(chi_contrib, 3)
        })

    S, q = 2, bins_count
    k = q - S - 1
    chi2_critical = chi2.ppf(1 - alpha, df=k)
    hypothesis_result = "принимается" if chi2_stat <= chi2_critical else "отвергается"

    df = pd.DataFrame(results)
    df.loc["Σ"] = ["–", "–", "–", N, "–", "–",
                   round(sum(r["p_j"] for r in results), 4),
                   round(sum(r["N*p_j"] for r in results), 2), "–", round(chi2_stat, 3)]
    print(df)
    print(f"\nСтатистика χ² = {chi2_stat:.3f}")
    print(f"Критическое значение χ²({k}, α={alpha}) = {chi2_critical:.2f}")
    print(f"Гипотеза о нормальном распределении {hypothesis_result}.\n")
    return df

def PlotHistograms():
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    samples = [x1, x2, n1, n2]
    labels = ["x1", "x2", "n1", "n2"]
    for i, (sample, label) in enumerate(zip(samples, labels)):
        for j, q in enumerate([7, 5]):
            axes[j, i].hist(sample, bins=q, edgecolor='black', alpha=0.7)
            axes[j, i].set_title(f"{label}, q={q}")
            axes[j, i].set_xlabel("Значения")
            axes[j, i].set_ylabel("Частота")
    plt.tight_layout()
    plt.savefig("histograms.png")
    plt.show()

for name, sample in zip(["x1", "x2", "n1", "n2"], [x1, x2, n1, n2]):
    PrintStatistics(name, ComputeStatistics(sample))

print("\nТаблицы для выборки x1:")
ComputeHistogram(x1, 7)
ComputeHistogram(x1, 5)
print("Таблицы для выборки x2:")
ComputeHistogram(x2, 7)
ComputeHistogram(x2, 5)
print("Таблицы для выборки n1:")
ComputeHistogram(n1, 7)
ComputeHistogram(n1, 5)
print("Таблицы для выборки n2:")
ComputeHistogram(n2, 7)
ComputeHistogram(n2, 5)

PlotHistograms()

print(f"\nУровень значимости: α = {alpha:.2f}")
for label, sample in zip(["x1", "x2", "n1", "n2"], [x1, x2, n1, n2]):
    for q in [7, 5]:
        print(f"Chi² ({label}, q={q}):")
        ChiSquareTable(sample, q)

