# utils/validate_rint_model.py
import os
import numpy as np
import matplotlib.pyplot as plt
import pickle
from tqdm import tqdm

# Best tuned Rint model parameters
BEST_PARAMS = {
    "R0": 0.0067,
    "R1": 0.1973,
    "C1": 5531.35,
    "k0": 28.6194,
    "k1": 0.1468,
    "k2": 0.0065
}

def rint_model_voltage(current, soc_init, time_step, params):
    R0, R1, C1 = params["R0"], params["R1"], params["C1"]
    k0, k1, k2 = params["k0"], params["k1"], params["k2"]
    eta = 0.99
    C_n = 22.0

    n = len(current)
    soc = soc_init
    V_rc = 0
    voltage = []

    for i in range(n):
        soc_clipped = np.clip(soc, 1e-6, 1 - 1e-6)
        ocv = k0 + k1 * np.log(soc_clipped) + k2 * np.log(1 - soc_clipped)
        V_rc = V_rc * np.exp(-time_step / (R1 * C1)) + current[i] * R1 * (1 - np.exp(-time_step / (R1 * C1)))
        v = ocv - current[i] * R0 - V_rc
        voltage.append(v)
        soc -= (eta / C_n) * current[i] * time_step
        soc = max(soc, 1e-6)

    return np.array(voltage)

if __name__ == "__main__":
    # Load training data
    with open("data/training_data.pkl", "rb") as f:
        data = pickle.load(f)

    subset_data = data[:5]
    time_step = 0.1
    soc_init = 1.0

    fig, axes = plt.subplots(len(subset_data), 1, figsize=(8, 2.5 * len(subset_data)), sharex=True)

    for idx, d in enumerate(tqdm(subset_data, desc="Validating Rint model")):
        current = d["input"]
        true_voltage = d["output"]
        time = d["time"]
        pred_voltage = rint_model_voltage(current, soc_init, time_step, BEST_PARAMS)

        rmse = np.sqrt(np.mean((true_voltage - pred_voltage)**2))

        axes[idx].plot(time, true_voltage, label="Actual Voltage", linewidth=2)
        axes[idx].plot(time, pred_voltage, "--", label=f"Rint Model (RMSE={rmse:.3f} V)", linewidth=2)
        axes[idx].set_ylabel("Voltage (V)")
        axes[idx].legend()
        axes[idx].grid(True)
        axes[idx].set_title(f"Sample {idx + 1}")

    axes[-1].set_xlabel("Time (s)")
    plt.tight_layout()
    os.makedirs("utils/figures", exist_ok=True)
    plt.savefig("utils/figures/validation_rint_comparison.pdf")
    plt.show()
