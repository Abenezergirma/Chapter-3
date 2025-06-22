import os
import numpy as np
import scipy.io
import matplotlib.pyplot as plt
import pickle
from pathlib import Path

# Configure Matplotlib
plt.rcParams["text.usetex"] = True

# ----------------------------
# Current Profile Generator Class
# ----------------------------
class CurrentProfileGenerator:
    """Converts power profiles to current profiles using battery SoC modeling."""
    
    def __init__(self, power_profile, soc_initial, battery_params):
        """Initializes the CurrentProfileGenerator class."""
        self.power_profile = np.array(power_profile)
        self.soc = soc_initial
        self.R_int = battery_params["R_int"]
        self.k0 = battery_params["k0"]
        self.k1 = battery_params["k1"]
        self.k2 = battery_params["k2"]
        self.C_n = battery_params["C_n"]
        self.eta = battery_params["eta"]
        self.current_profile = []

    def ocv_soc_curve(self, soc):
        """Computes the open-circuit voltage (OCV) using the Nernst model."""
        soc = np.clip(soc, 1e-6, 1 - 1e-6)  # Avoid log(0) errors
        return self.k0 + self.k1 * np.log(soc) + self.k2 * np.log(1 - soc)

    def solve_current(self, power, ocv):
        """Solves for current using the quadratic equation derived from the Rint model."""
        coefficients = [self.R_int, -ocv, power]
        roots_solution = np.roots(coefficients)

        # Convert complex numbers to real numbers by taking only the real part
        roots_solution = roots_solution.real

        # Select the smallest positive root
        current = np.min(roots_solution) if np.any(roots_solution >= 0) else np.min(roots_solution)

        return current

    def compute_current_profile(self):
        """Computes the current profile from the power profile using the Rint model."""
        if self.power_profile.shape == ():  # Empty case
            print("Skipping empty power profile...")
            return None  

        self.current_profile = []

        for power in self.power_profile:
            if power is None or np.isnan(power):  
                print("Skipping NaN or None value in power profile.")
                continue  

            ocv = self.ocv_soc_curve(self.soc)  
            current = self.solve_current(power, ocv)  
            self.current_profile.append(current)

            # Update SoC using Coulomb counting, ensuring it never goes negative
            self.soc = max(self.soc - (self.eta / self.C_n) * current, 1e-6)

        return np.array(self.current_profile)


# ----------------------------
# Power Profile Plotting Class
# ----------------------------
class PowerProfilePlotter:
    """Handles loading and visualization of power and current profiles."""

    def __init__(self, data_file, actual_simulation_file, battery_params, soc_initial=1.0):
        self.data_file = data_file
        self.actual_simulation_file = actual_simulation_file
        self.battery_params = battery_params
        self.soc_initial = soc_initial
        self.projected_profiles_list = self.load_data()
        self.load_actual_simulation_data()
        self.saved_current_profiles = []  

    def load_data(self):
        """Loads power profiles from a .mat file."""
        if not os.path.exists(self.data_file):
            print("Error: File not found.")
            return []
        data = scipy.io.loadmat(self.data_file, struct_as_record=False, squeeze_me=True, simplify_cells=True)
        return data.get("projected_profiles_list", [])

    def load_actual_simulation_data(self):
        """Loads actual simulation data for comparison plots."""
        if not os.path.exists(self.actual_simulation_file):
            print("Error: Actual simulation file not found.")
            self.timeb, self.totalCurrent, self.matlab_voltage = None, None, None
            return
        
        data = scipy.io.loadmat(self.actual_simulation_file, struct_as_record=False, squeeze_me=True, simplify_cells=True)
        self.timeb = data["results"][6]  
        self.totalCurrent = data["results"][1]  
        self.matlab_voltage = data["results"][2]  
    
    def plot_power_profiles(self):
        """
        Generates and saves a high-quality plot of power profiles,
        optimized for academic paper formatting in a single-column A4 layout.
        """
        if not self.projected_profiles_list:
            print("No projected profiles found.")
            return

        fig, ax = plt.subplots(figsize=(5, 4), dpi=300)
        colors = plt.cm.viridis(np.linspace(0, 1, len(self.projected_profiles_list)))
        time_shift = 0

        for idx, profile in enumerate(self.projected_profiles_list):
            ax.plot(profile['time'] + time_shift, profile['power'], color=colors[idx], linewidth=1.2, alpha=0.8)
        ax.plot(profile['time'] + time_shift, profile['power'], color=colors[idx], linewidth=1.2, alpha=0.8, label="Approx Power Profile")
        
        ax.plot(self.timeb + time_shift, self.totalCurrent*self.matlab_voltage, color='black', linewidth=1.2, label="Actual Power Profile")
        ax.set_xlabel('Time (s)', fontsize=12)
        ax.set_ylabel('Power (W)', fontsize=12)
        # ax.set_title('Projected Power Profiles', fontsize=14, fontweight='bold')
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # Create a colorbar instead of individual legends
        sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, norm=plt.Normalize(vmin=1, vmax=len(self.projected_profiles_list)))
        cbar = fig.colorbar(sm, ax=ax, aspect=30, pad=0.02)
        cbar.set_label("Profile Index", fontsize=10)
        ax.legend()
        base_dir = Path(__file__).resolve().parent.parent

        plt.savefig(base_dir  / "figures" / "Power_Profiles.pdf", format="pdf", bbox_inches="tight")
        plt.show()
        plt.close()
        
        # Compute RMSE and MAE between first approximated power profile (idx=0) and actual power profile
        approx_power = np.interp(self.timeb, self.projected_profiles_list[0]['time'], self.projected_profiles_list[0]['power'])
        actual_power = self.totalCurrent * self.matlab_voltage

        rmse = np.sqrt(np.mean((approx_power - actual_power) ** 2))
        mae = np.mean(np.abs(approx_power - actual_power))

        print(f"RMSE: {rmse:.4f} W")
        print(f"MAE: {mae:.4f} W")

    def plot_current_profiles(self):
        """Converts power profiles to current using battery modeling, plots them, and saves them."""
        if not self.projected_profiles_list:
            print("No projected profiles found.")
            return

        fig, ax = plt.subplots(figsize=(5, 4), dpi=300)
        colors = plt.cm.viridis(np.linspace(0, 1, len(self.projected_profiles_list)))

        first_valid_time_array = None  
        first_valid_current_profile = None  

        for idx, profile in enumerate(self.projected_profiles_list):
            power_profile = np.array(profile["power"])
            time_array = np.array(profile["time"])  

            # Skip empty or single-value profiles
            if power_profile.shape == () or power_profile.size == 1:
                print(f"Skipping profile {idx} due to invalid power data: {power_profile}")
                continue

            # Convert power to current using CurrentProfileGenerator
            current_generator = CurrentProfileGenerator(power_profile, self.soc_initial, self.battery_params)
            current_profile = current_generator.compute_current_profile()

            # Ensure time_array and current_profile have the same length
            min_length = min(len(time_array), len(current_profile))
            time_array = time_array[:min_length]  
            current_profile = current_profile[:min_length]  
            time_shift = 0

            # Store the first valid profile
            if first_valid_time_array is None and first_valid_current_profile is None:
                first_valid_time_array = time_array
                first_valid_current_profile = current_profile

            # Replace the first value only if there's enough data
            if idx != 0 and len(current_profile) > 1:
                current_profile[0] = current_profile[1]

            # Store the current profile for saving later
            self.saved_current_profiles.append({"time": time_array, "current": current_profile})

            ax.plot(time_array + time_shift, current_profile, color=colors[idx], alpha=0.8, linewidth=1.2)

        # Use the first valid time and current profile for final labeled plot
        if first_valid_time_array is not None and first_valid_current_profile is not None:
            ax.plot(first_valid_time_array + time_shift, first_valid_current_profile, color=colors[0], alpha=0.8, linewidth=1.2, label="Approx Current Profile")

        # Overlay actual current consumption from simulation
        if self.timeb is not None and self.totalCurrent is not None:
            ax.plot(self.timeb + time_shift, self.totalCurrent, color="black", linewidth=1, label="Actual Current Profile")

        ax.set_xlabel("Time (s)", fontsize=12)
        ax.set_ylabel("Current (A)", fontsize=12)
        ax.grid(True, linestyle="--", alpha=0.7)

        sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, norm=plt.Normalize(vmin=1, vmax=len(self.projected_profiles_list)))
        cbar = fig.colorbar(sm, ax=ax, aspect=30, pad=0.02)
        cbar.set_label("Profile Index", fontsize=10)
        base_dir = Path(__file__).resolve().parent.parent

        ax.legend()
        plt.savefig(base_dir / "figures" / "Current_Profiles.pdf", format="pdf", bbox_inches="tight")
        plt.show()
        plt.close()
        # Compute RMSE and MAE between first approximated current profile (idx=0) and actual power profile
        approx_current = np.interp(self.timeb, first_valid_time_array, first_valid_current_profile)
        actual_current = self.totalCurrent 

        rmse = np.sqrt(np.mean((approx_current - actual_current) ** 2))
        mae = np.mean(np.abs(approx_current - actual_current))

        print(f"RMSE: {rmse:.4f} A")
        print(f"MAE: {mae:.4f} A")

        # Save accumulated current profiles to a pickle file
        save_path = base_dir / "data" / "SimulationResults" / "saved_current_profiles.pkl"
        with open(save_path, "wb") as f:
            pickle.dump(self.saved_current_profiles, f)
        print("Current profiles saved to 'saved_current_profiles.pkl'.")

# ----------------------------
# Main Execution
# ----------------------------
if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent.parent  # data_driven_battery_modeling root
    projected_profiles_file = base_dir / "data" / "SimulationResults" / "projected_profiles_infeasible.mat"
    actual_simulation_file = base_dir / "data" / "SimulationResults" / "infeasibleTrajectorySimfullMissionBatteryParams.mat"


    battery_params = {"R_int": 0.135, "k0": 22.83, "k1": 0.39, "k2": -0.78, "C_n": 22, "eta": 0.99}

    plotter = PowerProfilePlotter(projected_profiles_file, actual_simulation_file, battery_params, soc_initial=1.0)
    plotter.plot_power_profiles()
    plotter.plot_current_profiles()
