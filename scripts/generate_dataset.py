import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.io import loadmat
import pickle
import random
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from pathlib import Path
from battery_model.battery_electrochem_TarotT18 import BatteryElectroChem as Battery

# ----------------------------
# Current Profile Generator Class
# ----------------------------
class CurrentProfileGenerator:
    def __init__(self, reference_profile, time_step=0.1):
        self.reference_profile = reference_profile
        self.total_length = len(reference_profile)
        self.time_step = time_step

    def generate_flight_mission_profile(self):
        """Generates a full flight mission profile with takeoff and cruise phases."""
        total_steps = self.total_length
        takeoff_duration_steps = np.random.randint(int(1 / self.time_step), int(3 / self.time_step) + 1)
        cruise_duration_steps = total_steps - takeoff_duration_steps

        takeoff_current = np.random.uniform(150, 240)
        cruise_current = np.random.uniform(50, 70)

        profile = np.zeros(total_steps)
        profile[:takeoff_duration_steps] = takeoff_current
        profile[takeoff_duration_steps:] = cruise_current

        time_horizon = np.arange(total_steps) * self.time_step
        # print(len(profile))
        # plt.plot(time_horizon, profile)
        # plt.show()
        return profile, time_horizon

    def generate_mid_flight_constant_profiles(self, flight_voltage_profile, flight_current_profile, step_interval=500):
        """Generates mid-flight current profiles based on actual voltage conditions."""
        total_steps = self.total_length
        mid_flight_profiles = []

        for start_time in range(step_interval, total_steps, step_interval):
            initial_voltage = flight_voltage_profile[start_time]  # Use actual voltage at mid-flight
            prior_current_profile = flight_current_profile[:start_time]  # Keep track of current before this timestamp
            remaining_current_profile = flight_current_profile[start_time:]  # Profile after the timestamp
            
            time_horizon = np.arange(start_time, total_steps) * self.time_step  

            mid_flight_profiles.append((prior_current_profile, remaining_current_profile, initial_voltage, time_horizon, start_time))

        return mid_flight_profiles

# ----------------------------
# Tarot Battery Class
# ----------------------------
class TarotBattery:
    def __init__(self, time_step):
        """Initializes the TarotT18 battery model with correct noise parameters."""
        R_vars = {"t": 2.2, "v": 1.46}
        self.battery = Battery(process_noise=0.25, process_noise_dist="normal",
                               measurement_noise=R_vars, measurement_noise_dist="normal")
        self.initial_state = self.battery.initialize()
        self.x = self.initial_state
        self.t = 0
        self.time_step = 0.1 #time_step

    def reset_battery_state(self, keep_voltage=False):
        """Resets the battery state while optionally preserving voltage for mid-flight profiles."""
        self.x = self.initial_state
        self.t = 0

    def set_initial_voltage(self, target_voltage, prior_current_profile, max_steps=100000):
        """
        Adjusts the battery state to reach the target voltage using the provided prior current profile.
        """
        self.reset_battery_state()  # ✅ Start from default state
        self.x = self.initial_state

        print(f"Before stabilization: Battery voltage = {self.battery.output(self.x)['v']:.2f}V")
        # print(f"Adjusting battery to initial voltage {target_voltage:.2f}V ...")

        for current in prior_current_profile:
            voltage = self.step(current)  # Use the prior current profile
            if voltage is None:
                raise ValueError("Battery model failed during stabilization.")
            if abs(voltage - target_voltage) < 0.001:  # ✅ Stop when close enough
                print(f"Reached {voltage:.2f}V after stabilization.")
                break

        print(f"⚡ After stabilization: Battery voltage = {self.battery.output(self.x)['v']:.2f}V")

    def step(self, current_needed, max_current=200):
        """Updates battery state given the current draw."""
        self.t += self.time_step
        try:
            current_clipped = min(current_needed, max_current)
            current_input = {"i": current_clipped}
            self.x = self.battery.next_state(self.x, current_input, self.time_step)
            output = self.battery.output(self.x)
            return output["v"]
        except (ValueError, KeyError) as e:
            print(f"Error during step execution: {e}")
            return None

    def simulate_battery(self, current_trajectory, initial_voltage=None, prior_current_profile=None):
        """Simulates battery voltage given a current trajectory."""
        if initial_voltage is not None and prior_current_profile is not None:
            self.set_initial_voltage(initial_voltage, prior_current_profile)

        voltage_trajectory = []
        for current in tqdm(current_trajectory, desc="Simulating Trajectory", leave=False):
            voltage = self.step(current)
            if voltage is None:
                return None
            voltage_trajectory.append(voltage)
        return np.array(voltage_trajectory)

# ----------------------------
# Main Script
# ----------------------------
if __name__ == "__main__":
    energy_req_path = Path(__file__).resolve().parent.parent / "data" / "EnergyReq.mat"
    energy_req = loadmat(str(energy_req_path))
    reference_current = energy_req["results"][1][1].flatten()

    profile_generator = CurrentProfileGenerator(reference_profile=reference_current)
    battery = TarotBattery(0.1)

    num_training = 70
    num_test = 30

    training_data = []
    test_data = []
    mid_flight_data = []
    full_flight_voltages = []
    full_flight_currents = []

    print("Generating training data...")
    for _ in tqdm(range(num_training), desc="Training Trajectories"):
        flight_profile, time_horizon = profile_generator.generate_flight_mission_profile()
        battery.reset_battery_state()
        voltage = battery.simulate_battery(flight_profile)
        if voltage is not None:
            training_data.append({"input": flight_profile, "output": voltage, "time": time_horizon})
            # plt.plot(time_horizon,voltage)
            # plt.show()
            full_flight_voltages.append(voltage)
            full_flight_currents.append(flight_profile)

    print("Generating test data...")
    for _ in tqdm(range(num_test), desc="Test Trajectories"):
        flight_profile, time_horizon = profile_generator.generate_flight_mission_profile()
        battery.reset_battery_state()
        voltage = battery.simulate_battery(flight_profile)
        if voltage is not None:
            test_data.append({"input": flight_profile, "output": voltage, "time": time_horizon})

    print("Generating mid-flight feasibility check profiles...")
    for voltage_trajectory, current_trajectory in tqdm(zip(full_flight_voltages, full_flight_currents), desc="Processing Full-Flight Profiles"):
        mid_flight_profiles = profile_generator.generate_mid_flight_constant_profiles(voltage_trajectory, current_trajectory)

        for prior_current_profile, remaining_current_profile, initial_voltage, time_horizon, start_time in tqdm(mid_flight_profiles, desc="Processing Mid-Flight Profiles", leave=False):
            battery.reset_battery_state(keep_voltage=False)
            battery.set_initial_voltage(initial_voltage, prior_current_profile)

            voltage = battery.simulate_battery(remaining_current_profile, initial_voltage, prior_current_profile)
            if voltage is not None:
                mid_flight_data.append({
                    "input": remaining_current_profile,
                    "output": voltage,
                    "time": time_horizon
                })
    
    # Compute split indices
    mid_flight_train_size = int(0.6 * len(mid_flight_data))  # 70% for training
    mid_flight_test_size = len(mid_flight_data) - mid_flight_train_size  # 30% for testing

    # Split the mid-flight data
    mid_flight_train = mid_flight_data[:mid_flight_train_size]
    mid_flight_test = mid_flight_data[mid_flight_train_size:]

    # Extend existing training and test data
    training_data.extend(mid_flight_train)
    test_data.extend(mid_flight_test)

    # Ensure directories exist
    os.makedirs("data", exist_ok=True)
    os.makedirs("figures", exist_ok=True)

    # Save training and test data
    with open("data/training_data_mixed_70.pkl", "wb") as f:
        pickle.dump(training_data, f)

    with open("data/test_data_mixed_30.pkl", "wb") as f:
        pickle.dump(test_data, f)

    print(f" Saved training data to data/training_data.pkl")
    print(f" Saved test data to data/test_data.pkl")

    print(f"Training data size after adding mid-flight samples: {len(training_data)}")
    print(f"Test data size after adding mid-flight samples: {len(test_data)}")
    print("Training and test datasets updated successfully.")

    def plot_combined_results(training_data, test_data, mid_flight_data, filename="figures/combined_results.png"):
        """Saves and visualizes all generated current and voltage profiles."""
        fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=120)

        for data in training_data:
            axes[0].plot(data["time"], data["input"], color="blue", linewidth=0.8, alpha=0.6)
            axes[1].plot(data["time"], data["output"], color="blue", linewidth=0.8, alpha=0.6)

        for data in test_data:
            axes[0].plot(data["time"], data["input"], color="red", linewidth=0.8, alpha=0.6)
            axes[1].plot(data["time"], data["output"], color="red", linewidth=0.8, alpha=0.6)

        for data in mid_flight_data:
            axes[0].plot(data["time"], data["input"], color="black", linewidth=0.8, alpha=0.8)
            axes[1].plot(data["time"], data["output"], color="black", linewidth=0.8, alpha=0.8)

        axes[0].set_title("Current Trajectories")
        axes[0].set_ylabel("Current (A)")
        axes[0].set_xlabel("Time (s)")
        axes[1].set_title("Voltage Responses")
        axes[1].set_ylabel("Voltage (V)")
        axes[1].set_xlabel("Time (s)")

        plt.tight_layout()
        plt.savefig(filename)
        print(f"Combined results saved to {filename}")
        plt.show()

    plot_combined_results(training_data, test_data, mid_flight_data)
