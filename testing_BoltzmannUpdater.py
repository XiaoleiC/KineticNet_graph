from model_Boltzmann import BoltzmannUpdater
import torch

N = 3
Q_mesoscale = 6
min_macrovelocity = 5
max_macrovelocity = 20

xi_velocity_per_node = torch.tensor([
    [10, 11, 12, 13, 14, 15],  # node 0: [10-15]
    [8, 9, 10, 11, 12, 13],    # node 1: [8-13] 
    [15, 16, 17, 18, 19, 20],  # node 2: [15-20]
])

updater = BoltzmannUpdater(
    Q_mesoscale=Q_mesoscale,
    min_macrovelocity=min_macrovelocity,
    max_macrovelocity=max_macrovelocity,
    xi_velocity_per_node=xi_velocity_per_node
)

f_local = torch.rand(N, Q_mesoscale)
# print(f"Local shape: {f_local.shape}")
# print(f"Local data for node 1: {f_local[1]}")

f_unified = updater.convert_to_unified(f_local)

for i in range(f_unified.shape[0]):
    print(f"Unified data for node {i}: {f_unified[i]}")
    print(f'original xi_velocity for node {i}: {f_local[i]}')

