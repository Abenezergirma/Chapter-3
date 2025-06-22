import numpy as np
import pickle
from scipy.optimize import differential_evolution
from tqdm import tqdm

# ----------- Normalization Helpers -----------
def normalize(data, min_val, max_val):
    return (data - min_val) / (max_val - min_val)

def denormalize(data, min_val, max_val):
    return data * (max_val - min_val) + min_val

# ----------- Load Training Data -----------
with open("data/training_data.pkl", "rb") as f:
    training_data = pickle.load(f)

voltage_actual_list = [training_data[i]["output"] for i in range(min(100, len(training_data)))]
voltage_min = min(v.min() for v in voltage_actual_list)
voltage_max = max(v.max() for v in voltage_actual_list)

# ----------- Two-Electrode Nernst Loss Function -----------
def two_electrode_nernst_loss(params):
    U0n, U0p, a, b = params
    total_loss = 0.0

    for voltage_actual in voltage_actual_list:
        voltage_norm = normalize(voltage_actual, voltage_min, voltage_max)
        x_conc = np.clip(a * voltage_norm + b, 1e-4, 1 - 1e-4)

        voltage_pred_norm = U0p - U0n + np.log((1 - x_conc) / x_conc)
        voltage_pred = denormalize(voltage_pred_norm, voltage_min, voltage_max)

        rmse = np.sqrt(np.mean((voltage_pred - voltage_actual) ** 2))
        total_loss += rmse

    return total_loss / len(voltage_actual_list)

# ----------- Global Optimization using Differential Evolution -----------
bounds = [(0.01, 0.2), (23.5, 25.5), (0.5, 1.5), (-0.5, 0.5)]  # Bounds for U0n, U0p, a, b

print("🔧 Optimizing Two-Electrode Nernst parameters with a, b mapping...")
result = differential_evolution(two_electrode_nernst_loss, bounds, maxiter=200, tol=1e-6, polish=True, seed=42)

# ----------- Report Results -----------
best_U0n, best_U0p, best_a, best_b = result.x
best_rmse = result.fun

print("\n✅ Two-Electrode Nernst Parameter Tuning Complete (with flexible concentration mapping):")
print(f"📉 Best Average RMSE: {best_rmse:.4f} V")
print(f"🔧 Optimized Parameters:")
print(f"   U0n = {best_U0n:.4f} V")
print(f"   U0p = {best_U0p:.4f} V")
print(f"   a   = {best_a:.4f}")
print(f"   b   = {best_b:.4f}")
