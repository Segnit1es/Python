import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

np.random.seed(0)
N = 100
Mb = 5.0
x = np.random.exponential(scale=1/Mb, size=N)
sum_x = np.sum(x)

lambda_vals = np.linspace(0.1, 10, 180)
sigma_vals = np.linspace(0.1, 2, 220)
Lambda_grid, Sigma_grid = np.meshgrid(lambda_vals, sigma_vals, indexing='ij')

logL = N * np.log(Lambda_grid) - Lambda_grid * sum_x

idx = np.unravel_index(np.argmax(logL), logL.shape)
lambda_max, sigma_max = lambda_vals[idx[0]], sigma_vals[idx[1]]
L_max = logL[idx]

print(f"[1] Максимум logL: λ = {lambda_max:.4f}, σ = {sigma_max:.4f}, logL = {L_max:.4f}")
print(f"Истинное значение λ = {Mb:.1f}")


fig = plt.figure(figsize=(12, 5))

ax1 = fig.add_subplot(1, 2, 1, projection='3d')
ax1.plot_surface(Lambda_grid, Sigma_grid, logL, cmap='viridis', edgecolor='none', alpha=0.9)
ax1.set_title('3D-график logL(lgLambdda, σ)')
ax1.set_xlabel('lgLambda')
ax1.set_ylabel('σ')
ax1.set_zlabel('logL')

ax2 = fig.add_subplot(1, 2, 2)
cs = ax2.contourf(lambda_vals, sigma_vals, logL.T, levels=30, cmap='viridis')
ax2.plot(np.log10(lambda_max), sigma_max, 'ro', label=f'максимум (λ={lambda_max:.2f})')
ax2.set_title('Контур logL(lgLambda, σ)')
ax2.set_xlabel('lgLambda')
ax2.set_ylabel('σ')
plt.colorbar(cs, ax=ax2)
ax2.legend()

plt.tight_layout()
plt.show()

