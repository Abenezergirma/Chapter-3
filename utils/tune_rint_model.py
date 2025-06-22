import numpy as np
import pickle
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from tqdm import tqdm

# ----------------------------
# RC Battery Model Definition
# ----------------------------
class RCBatteryModel:
    def __init__(self, R0, R1, C1, k0, k1, k2, C_n=22, eta=0.99, time_step=0.1):
        self.R0 = R0
        self.R1 = R1
        self.C1 = C1
        self.k0 = k0
        self.k1 = k1
        self.k2 = k2
        self.C_n = C_n
        self.eta = eta
        self.dt = time_step

    def ocv(self, soc):
        soc = np.clip(soc, 1e-6, 1 - 1e-6)
        return self.k0 + self.k1 * np.log(soc) + self.k2 * np.log(1 - soc)

    def simulate(self, current_profile):
        soc = 1.0
        V_RC = 0.0
        voltages = []

        for I in current_profile:
            V_OCV = self.ocv(soc)
            V_terminal = V_OCV - I * self.R0 - V_RC
            voltages.append(V_terminal)

            dV_RC = (-V_RC / (self.R1 * self.C1) + I / self.C1) * self.dt
            V_RC += dV_RC
            soc = max(soc - (self.eta / self.C_n) * I * self.dt, 1e-6)

        return np.array(voltages)

# ----------------------------
# Objective Function for One Profile
# ----------------------------
def objective(params, current, actual_voltage):
    R0, R1, C1, k0, k1, k2 = params
    model = RCBatteryModel(R0, R1, C1, k0, k1, k2)
    predicted_voltage = model.simulate(current)
    rmse = np.sqrt(np.mean((predicted_voltage - actual_voltage) ** 2))
    return rmse

# ----------------------------
# Load and Optimize Across Multiple Profiles
# ----------------------------
def main():
    with open("data/training_data.pkl", "rb") as f:
        training_data = pickle.load(f)

    subset_data = training_data[:5]  # Use first few for faster fitting

    def combined_objective(params):
        total_rmse = 0
        desc = f"Evaluating RMSE for Params: R0={params[0]:.4f}, R1={params[1]:.4f}, C1={params[2]:.2f}"
        for d in tqdm(subset_data, desc=desc, leave=False):
            current = d["input"]
            voltage = d["output"]
            rmse = objective(params, current, voltage)
            total_rmse += rmse
        return total_rmse / len(subset_data)

    initial_params = [0.135, 0.01, 500.0, 22.8, 0.4, -0.8]  # R0, R1, C1, k0, k1, k2

    print("🔧 Optimizing RC Battery Model Parameters...")
    result = minimize(combined_objective, initial_params, method='Nelder-Mead')
    optimized_params = result.x
    final_rmse = result.fun

    print(f"\n📉 Best Average RMSE: {final_rmse:.4f} V")
    print("🔧 Optimized Parameters:")
    print(f"  R0 = {optimized_params[0]:.4f} Ω")
    print(f"  R1 = {optimized_params[1]:.4f} Ω")
    print(f"  C1 = {optimized_params[2]:.2f} F")
    print(f"  k0 = {optimized_params[3]:.4f}, k1 = {optimized_params[4]:.4f}, k2 = {optimized_params[5]:.4f}")

    # ----------------------------
    # Plot Fit
    # ----------------------------
    model = RCBatteryModel(*optimized_params)
    plt.figure(figsize=(10, 6))

    for idx, d in enumerate(subset_data):
        current = d["input"]
        voltage_actual = d["output"]
        voltage_pred = model.simulate(current)

        plt.plot(d["time"], voltage_actual, label=f"Actual {idx+1}", linestyle='--')
        plt.plot(d["time"], voltage_pred, label=f"RC Pred {idx+1}")

    plt.title(f"RC Model Voltage Fit (Avg RMSE = {final_rmse:.4f} V)")
    plt.xlabel("Time (s)")
    plt.ylabel("Voltage (V)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("utils/RC_voltage_fitting_result.pdf")
    plt.show()

if __name__ == "__main__":
    main()
