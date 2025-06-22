# Re-import necessary libraries since execution state was reset
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
plt.rcParams['text.usetex'] = True

# Given SoC values (0-1 scale)
soc_values = np.array([100, 95, 90, 85, 80, 75, 70, 65, 60, 55, 50, 45, 40, 35, 30, 25, 20, 15, 10, 5, 0]) / 100

# Corresponding OCV values for 6S pack (V)
voltage_pack = np.array([25.20, 24.90, 24.66, 24.48, 24.12, 23.88, 23.70, 23.46, 23.22, 23.10, 
                         23.04, 22.92, 22.80, 22.74, 22.62, 22.50, 22.38, 22.26, 22.14, 21.66, 19.62])

# Define the Nernst-based empirical model
def nernst_model(soc, k0, k1, k2):
    return k0 + k1 * np.log(soc) + k2 * np.log(1 - soc)

# Fit the model to the data (excluding SoC = 0 and SoC = 1 to avoid log(0))
initial_guess = [24, -2, -2]  # Initial guess for k0, k1, k2
params, covariance = curve_fit(nernst_model, soc_values[1:-1], voltage_pack[1:-1], p0=initial_guess)
print(params)

# Generate smooth curve
soc_smooth = np.linspace(0.01, 0.99, 100)  # Avoiding log(0)
ocv_smooth = nernst_model(soc_smooth, *params)

# Plot the data and fitted curve
plt.figure(figsize=(5, 4))
plt.scatter(soc_values * 100, voltage_pack, color='red', label="Battery Data", marker='x')
plt.plot(soc_smooth * 100, ocv_smooth, 'b-', label="OCV-SoC Curve")

# Labels and title
plt.xlabel("State of Charge (\%)")
plt.ylabel("Open Circuit Voltage (V)")
# plt.title("Fitted SoC vs OCV Curve for Lumenier 6S1P 22Ah LiPo Battery")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.5)
plt.savefig("soc_ocv.pdf")


# Display the plot
plt.show()

# Display fitted parameters
params
