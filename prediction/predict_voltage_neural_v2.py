import os
import pickle
import numpy as np
import scipy.interpolate as interp
from scipy.interpolate import interp1d
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torchdiffeq import odeint
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
import sys
import time 
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from scripts.generate_dataset import TarotBattery


# sys.path.append(str(Path(__file__).resolve().parents[1]))
# from scripts.generate_dataset import TarotBattery  # Adjust the import path as necessary

plt.rcParams["text.usetex"] = True

def extrapolate_array(arr, new_length):
    """ Extrapolates an array to a new length based on linear trends of the last segment. """
    x_old = np.arange(len(arr))
    f = interp1d(x_old, arr, kind='linear', fill_value='extrapolate')
    x_new = np.linspace(0, len(arr)-1, new_length)
    return f(x_new)

def standardize(data, times, full_mean, full_std, training_time):
    data = np.asarray(data)
    times = np.asarray(times)
    max_index = len(full_mean) - 1
    if times.ndim == 0:
        index = np.round(np.interp(times, training_time, np.arange(len(training_time)))).astype(int)
        index = min(index, max_index)
        return (data - full_mean[index]) / full_std[index]
    else:
        indices = np.round(np.interp(times, training_time, np.arange(len(training_time)))).astype(int)
        indices = np.clip(indices, 0, max_index)
        if max(indices) < len(data) - 1 and len(times) == len(data):
            new_times = times
            interp_mean = np.interp(new_times, training_time, full_mean)
            interp_std = np.interp(new_times, training_time, full_std)
            return (data - interp_mean) / interp_std
        else:
            return (data - full_mean[indices]) / full_std[indices]

def destandardize(data, times, full_mean, full_std, training_time):
    data = np.asarray(data)
    times = np.asarray(times)
    training_time = np.asarray(training_time)
    full_mean = np.asarray(full_mean)
    full_std = np.asarray(full_std)

    # Ensure 1D for interpolation safety
    if times.ndim == 0:
        times = np.array([times])
        scalar_input = True
    else:
        scalar_input = False

    # Compute slope from last two points for extrapolation
    def extrapolate(arr):
        slope = arr[-1] - arr[-2]
        def extrapolated_fn(t):
            return arr[-1] + slope * ((t - training_time[-1]) / (training_time[-1] - training_time[-2]))
        return extrapolated_fn

    mean_extrap_fn = extrapolate(full_mean)
    std_extrap_fn = extrapolate(full_std)

    # Initialize arrays for interpolated or extrapolated values
    interp_mean = np.empty_like(times, dtype=float)
    interp_std = np.empty_like(times, dtype=float)

    for i, t in enumerate(times):
        if t <= training_time[-1]:
            interp_mean[i] = np.interp(t, training_time, full_mean)
            interp_std[i] = np.interp(t, training_time, full_std)
        else:
            interp_mean[i] = mean_extrap_fn(t)
            interp_std[i] = std_extrap_fn(t)

    result = data * interp_std + interp_mean
    return result[0] if scalar_input else result

        


def load_data(training_file, test_file):
    with open(training_file, "rb") as f:
        training_data = pickle.load(f)
    with open(test_file, "rb") as f:
        test_data = pickle.load(f)

    all_inputs = np.array([d["input"] for d in training_data])
    all_outputs = np.array([d["output"] for d in training_data])

    input_mean = np.mean(all_inputs, axis=0)
    input_std = np.std(all_inputs, axis=0)
    output_mean = np.mean(all_outputs, axis=0)
    output_std = np.std(all_outputs, axis=0)

    input_std[input_std == 0] = 1
    output_std[output_std == 0] = 1

    return (input_mean, input_std), (output_mean, output_std)

class ODEFunc(nn.Module):
    def __init__(self, state_dim=1, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        self.current_profile = None
        self.time_tensor = None

    def set_current_profile(self, current_profile):
        self.current_profile = current_profile

    def set_time_tensor(self, t):
        self.time_tensor = t

    def forward(self, t, x):
        if self.current_profile is None or self.time_tensor is None:
            raise ValueError("Current profile or time tensor not set.")

        batch_size, seq_len, _ = self.current_profile.shape
        t_all = self.time_tensor
        t_scalar = t.item()
        time_idx = torch.argmin(torch.abs(t_all - t_scalar)).item()
        current_at_t = self.current_profile[:, time_idx, :]
        time_tensor = t.expand(batch_size, 1)
        input_tensor = torch.cat((x, current_at_t, time_tensor), dim=-1)
        return self.net(input_tensor)

class NeuralODE(nn.Module):
    def __init__(self, state_dim, hidden_dim):
        super(NeuralODE, self).__init__()
        self.ode_func = ODEFunc(state_dim, hidden_dim)

    def forward(self, x0, current_profile, t):
        self.ode_func.set_current_profile(current_profile)
        self.ode_func.set_time_tensor(t)
        out = odeint(self.ode_func, x0, t, method="rk4", rtol=1e-3, atol=1e-4)
        return out.permute(1, 0, 2)

class VoltagePredictor:
    def __init__(self, model_path, file_path, training_file, test_file):
        self.model_path = model_path
        self.current_profiles = self.load_saved_current_profiles(file_path)
        self.current_range, self.voltage_range = load_data(training_file, test_file)
        self.current_mean, self.current_std = self.current_range
        self.voltage_mean, self.voltage_std = self.voltage_range
        self.ode_func = ODEFunc()
        self.neural_ode = NeuralODE( state_dim=1, hidden_dim=128)
        self.neural_ode.load_state_dict(torch.load(self.model_path))
        self.neural_ode.eval()
        self.clipped_current_profiles = None 
        # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Infer training time assuming dt is known
        dt = 0.1  # Assume time step is 0.1 seconds; adjust this based on actual data properties
        self.training_time = np.linspace(0, len(self.current_mean) * dt - dt, len(self.current_mean))
        self.battery_simulator = TarotBattery(time_step=0.1)


    def load_saved_current_profiles(self, file_path):
        if not os.path.exists(file_path):
            print("Error: Saved current profiles file not found.")
            return []
        with open(file_path, "rb") as f:
            return pickle.load(f)


    def predict_voltage(self):
        predicted_voltage_trajectories = []
        max_length = len(self.current_profiles[0]["current"])  # Length of the first current trajectory
        self.clipped_current_profiles = []  # Storing clipped current profiles

        # Initial voltage for the first prediction
        initial_voltage_standardized = 0.0
        x0 = torch.tensor([[initial_voltage_standardized]], dtype=torch.float32)

        for idx, current_profile in enumerate(self.current_profiles):
            # print(current_profile)
            start_time = time.time()

            # Clipping current profiles
            current_values = current_profile["current"][:max_length]
            times = current_profile["time"][:max_length]
            self.clipped_current_profiles.append({"current": current_values, "time": times})

            standardized_current = standardize(current_values, times, self.current_mean, self.current_std, self.training_time)
            current_tensor = torch.tensor(standardized_current, dtype=torch.float32).unsqueeze(0).unsqueeze(-1)

            t = torch.linspace(0, 1, len(current_values))  # Assuming uniform time steps
            self.ode_func.set_current_profile(current_tensor)
            self.ode_func.set_time_tensor(t)
            pred_output = self.neural_ode(x0, current_tensor, t).squeeze(0).squeeze(-1)
            voltage = destandardize(pred_output.detach().numpy(), times, self.voltage_mean, self.voltage_std, self.training_time)

            # Store voltage trajectory for visualization or further use
            predicted_voltage_trajectories.append({"time": times, "voltage": voltage})

            # Print and plot results
            print(f"Prediction {idx}: Time taken = {time.time() - start_time:.2f} seconds")
            # plt.plot(times, voltage)
            # plt.show()

            if idx < len(self.current_profiles) - 1:
                # Prepare initial voltage for the next prediction
                next_initial_time = self.current_profiles[idx + 1]['time'][0]
                # print(times.shape)
                time_index = np.searchsorted(times, next_initial_time)
                initial_voltage_standardized = pred_output[time_index] #- self.voltage_mean[0]) / self.voltage_std[0]
                x0 = torch.tensor([[initial_voltage_standardized]], dtype=torch.float32)

        return predicted_voltage_trajectories



    def predict_voltage_new(self):
        predicted_voltage_trajectories = []
        self.clipped_current_profiles = []
        max_length = len(self.current_profiles[0]["current"])

        # Simulate first profile for initial voltages
        first_profile = self.current_profiles[0]
        actual_start_time = first_profile['time'][0]
        if actual_start_time > 0:
            dummy_time = np.arange(0, actual_start_time + 5, 5)
            if dummy_time[-1] < actual_start_time:
                dummy_time = np.append(dummy_time, actual_start_time)
            dummy_current = np.full_like(dummy_time, 64.6)
            combined_time = np.concatenate([dummy_time, first_profile['time']])
            combined_current = np.concatenate([dummy_current, first_profile['current']])
        else:
            combined_time = first_profile['time'][:max_length]
            combined_current = first_profile['current'][:max_length]

        # Interpolate to 0.1-second step
        new_combined_time = np.arange(combined_time[0], combined_time[-1] + 0.1, 0.1) if len(combined_time) > 1 else combined_time
        new_combined_current = np.interp(new_combined_time, combined_time, combined_current) if len(combined_time) > 1 else combined_current

        # Simulate battery
        # self.battery_simulator.set_initial_voltage(27.4, new_combined_current)
        simulated_voltage = self.battery_simulator.simulate_battery(new_combined_current, 27.4)
        simulated_time = new_combined_time

        # Sample initial voltages every 50 points
        clip_idx = np.searchsorted(simulated_time, actual_start_time)
        simulated_voltage = simulated_voltage[clip_idx:]
        simulated_time = simulated_time[clip_idx:]
        sample_indices = np.arange(0, len(simulated_voltage), 50)
        num_samples = min(len(sample_indices), len(self.current_profiles))

        initial_voltages = []
        for i in range(len(self.current_profiles)):
            sample_idx = sample_indices[min(i, num_samples - 1)]
            voltage = simulated_voltage[sample_idx]
            standardized_voltage = standardize(voltage, simulated_time[sample_idx], 
                                            self.voltage_mean, self.voltage_std, 
                                            self.training_time)
            initial_voltages.append(float(standardized_voltage))

        index_counter = 0
        for idx, profile in enumerate(self.current_profiles):
            start_time = time.time()

            # Create combined profile
            actual_start_time = profile['time'][0]
            if actual_start_time > 0:
                dummy_time = np.arange(0, actual_start_time + 5, 5)
                if dummy_time[-1] < actual_start_time:
                    dummy_time = np.append(dummy_time, actual_start_time)
                dummy_current = np.full_like(dummy_time, 64.6)
                combined_time = np.concatenate([dummy_time, profile['time']])
                combined_current = np.concatenate([dummy_current, profile['current']])
            else:
                combined_time = profile['time'][:max_length]
                combined_current = profile['current'][:max_length]

            # Interpolate to 0.1-second step
            new_times_np = np.arange(combined_time[0], combined_time[-1] + 0.1, 0.1) if len(combined_time) > 1 else combined_time
            new_current_np = np.interp(new_times_np, combined_time, combined_current) if len(combined_time) > 1 else combined_current

            # Save interpolated profile
            clip_idx = np.searchsorted(new_times_np, actual_start_time)
            self.clipped_current_profiles.append({
                "current": new_current_np[clip_idx:],
                "time": new_times_np[clip_idx:]
            })

            # Convert to tensors
            times = torch.tensor(new_times_np, dtype=torch.float32)
            current_values = torch.tensor(new_current_np, dtype=torch.float32)

            # Standardize current
            standardized_current = torch.tensor(
                standardize(current_values.cpu().numpy(), new_times_np, 
                            self.current_mean, self.current_std, 
                            self.training_time),
                dtype=torch.float32
            ).unsqueeze(0).unsqueeze(-1)

            # Neural ODE prediction
            with torch.no_grad():
                t = torch.linspace(0, 1, len(new_current_np))
                self.ode_func.set_current_profile(standardized_current)
                self.ode_func.set_time_tensor(t)
                x0 = torch.tensor([[initial_voltages[idx]]], dtype=torch.float32)
                pred_output = self.neural_ode(x0, standardized_current, t).squeeze(0).squeeze(-1)
                voltage = destandardize(pred_output.cpu().numpy(), new_times_np, 
                                        self.voltage_mean, self.voltage_std, 
                                        self.training_time)

            # Clip and interpolate voltage trajectory
            clipped_time = new_times_np[clip_idx:]
            clipped_voltage = voltage[clip_idx:]
            new_clipped_time = np.arange(clipped_time[0], clipped_time[-1] + 0.1, 0.1) if len(clipped_time) > 1 else clipped_time
            new_clipped_voltage = np.interp(new_clipped_time, clipped_time, clipped_voltage) if len(clipped_time) > 1 else clipped_voltage

            predicted_voltage_trajectories.append({
                "time": new_clipped_time,
                "voltage": new_clipped_voltage
            })

            # Update initial voltage for next profile
            if idx < len(self.current_profiles) - 1:
                next_initial_time = self.current_profiles[idx + 1]['time'][0]
                time_index = np.searchsorted(new_times_np, next_initial_time)
                if time_index >= len(pred_output):
                    time_index = len(pred_output) - 1
                index_counter += time_index
                if index_counter >= len(simulated_voltage):
                    index_counter = len(simulated_voltage) - 1
                initial_voltage = standardize(
                    simulated_voltage[index_counter], simulated_time[index_counter], 
                    self.voltage_mean, self.voltage_std, 
                    self.training_time)
                initial_voltages[idx + 1] = float(initial_voltage)

            print(f"Prediction {idx}: Time taken = {time.time() - start_time:.2f} seconds")

        # Save profiles
        with open("new_voltages/saved_clipped_current_profiles.pkl", "wb") as f:
            pickle.dump(self.clipped_current_profiles, f)
        print("clipped_current_profiles saved to 'saved_clipped_current_profiles.pkl'.")

        with open("new_voltages/saved_predicted_voltage_trajectories.pkl", "wb") as f:
            pickle.dump(predicted_voltage_trajectories, f)
        print("predicted_voltage_trajectories saved to 'saved_predicted_voltage_trajectories.pkl'.")

        return predicted_voltage_trajectories

    def simulate_actual_battery(self, predicted_voltage_trajectories):
        """Simulates actual battery behavior using saved current profiles."""
        if not self.clipped_current_profiles:
            print("No saved current profiles found.")
            return []

        actual_voltage_trajectories = []
        for idx, (current_profile, predicted) in enumerate(zip(self.clipped_current_profiles, predicted_voltage_trajectories)):
            time = np.array(current_profile["time"])
            current = np.array(current_profile["current"])
            predicted_initial_voltage = predicted["voltage"][0]
            # print(predicted_initial_voltage.shape)
            # print(time.shape)
            # print(current.shape)
            # print("predicted", predicted_initial_voltage)
            

            self.battery_simulator.set_initial_voltage(predicted_initial_voltage, current)
            voltage_trajectory = self.battery_simulator.simulate_battery(current, predicted_initial_voltage) - 1.609788619596788*0
            # print("actual", voltage_trajectory[0])
            # print("diff is", voltage_trajectory[0] - predicted_initial_voltage)
            # plt.plot(predicted["voltage"])
            # plt.plot(voltage_trajectory)
            # plt.show()

            if voltage_trajectory is not None:
                actual_voltage_trajectories.append({"time": time.tolist(), "voltage": voltage_trajectory.tolist()})
                
        with open("new_voltages/saved_actual_voltage_trajectories.pkl", "wb") as f:
            pickle.dump(actual_voltage_trajectories, f)
        print("actual_voltage_trajectories saved to 'actual_voltage_trajectories.pkl'.")

        return actual_voltage_trajectories



    def plot_comparison_voltage_trajectories(self, predicted_voltage_trajectories, actual_voltage_trajectories):
        fig, ax = plt.subplots(figsize=(5, 4), dpi=300)
        num_trajectories = min(50, len(predicted_voltage_trajectories))
        colors = plt.cm.viridis(np.linspace(0, 1, num_trajectories))
        time_shift = 0
        batt_calibration = 2

        for idx, (predicted, actual) in enumerate(zip(predicted_voltage_trajectories[:50], actual_voltage_trajectories[:50])):
            pred_time, pred_voltage = np.array(predicted["time"]), np.array(predicted["voltage"])  + 0.318*0
            actual_time, actual_voltage = np.array(actual["time"]), np.array(actual["voltage"])

            # Validate arrays
            if pred_time.size == 0 or pred_voltage.size == 0 or actual_time.size == 0 or actual_voltage.size == 0:
                print(f"Skipping trajectory {idx+1} due to empty arrays.")
                continue
            if pred_time.shape != pred_voltage.shape or actual_time.shape != actual_voltage.shape:
                print(f"Skipping trajectory {idx+1} due to shape mismatch: "
                    f"pred_time shape={pred_time.shape}, pred_voltage shape={pred_voltage.shape}, "
                    f"actual_time shape={actual_time.shape}, actual_voltage shape={actual_voltage.shape}")
                continue

            # Interpolate predicted voltage to match actual time points
            try:
                pred_voltage_interp = np.interp(actual_time, pred_time, pred_voltage)
            except ValueError as e:
                print(f"Skipping trajectory {idx+1} due to interpolation error: {e}")
                continue

            # Compute RMSE and MAE
            rmse = np.sqrt(np.mean((pred_voltage_interp - actual_voltage) ** 2))
            mae = np.mean(np.abs(pred_voltage_interp - actual_voltage))
            print(f"Trajectory {idx+1} Voltage RMSE: {rmse:.4f} V")
            print(f"Trajectory {idx+1} Voltage MAE: {mae:.4f} V")

            # Plot trajectories
            ax.plot(pred_time + time_shift, pred_voltage - batt_calibration, 
                    linestyle="dashed", color=colors[idx], linewidth=0.8, alpha=0.7)
            # ax.plot(actual_time + time_shift, actual_voltage - batt_calibration, 
            #         linestyle="solid", color=colors[idx], linewidth=0.8, alpha=0.7)

        # Plot the first trajectory with labels for legend
        if num_trajectories > 0:
            first_pred = predicted_voltage_trajectories[0]  
            first_actual = actual_voltage_trajectories[0]
            ax.plot(first_pred["time"] + time_shift, np.array(first_pred["voltage"]) - batt_calibration + 0.318*0, 
                    linestyle="dashed", color=colors[0], linewidth=0.8, alpha=0.7, label="Predicted Voltage")
            ax.plot(first_actual["time"] , np.array(first_actual["voltage"]) - batt_calibration, 
                    linestyle="solid", color=colors[0], linewidth=0.8, alpha=0.7, label="Actual Voltage")

        # Add voltage threshold line at 18V
        ax.axhline(y=18, color='red', linestyle='dashed', linewidth=1.5, label="Voltage Threshold")

        # Set labels and grid
        ax.set_xlabel("Time (s)", fontsize=12)
        ax.set_ylabel("Voltage (V)", fontsize=12)
        ax.grid(True, linestyle="--", alpha=0.7)

        # Add colorbar for trajectory index
        if num_trajectories > 0:
            sm = ScalarMappable(cmap=plt.cm.viridis, norm=Normalize(vmin=1, vmax=num_trajectories))
            cbar = fig.colorbar(sm, ax=ax, aspect=30, pad=0.02)
            cbar.set_label("Trajectory Index", fontsize=10)

        # Add legend
        ax.legend(loc="best", fontsize=10, frameon=False)

        # Save and show plot
        plt.savefig(BASE_DIR / "results" / "Voltage_Comparison.pdf", format="pdf", bbox_inches="tight")
        plt.show()
        plt.close()


    def plot_comparison_voltage_trajectories_old(self, predicted_voltage_trajectories, actual_voltage_trajectories):
        """Plots both predicted and actual battery voltage trajectories for comparison.
        Adds a voltage threshold line, a mid-time indicator, and computes RMSE & MAE.
        Keeps the legend inside but makes it compact.
        """
        fig, ax = plt.subplots(figsize=(5, 4), dpi=300)
        colors = plt.cm.viridis(np.linspace(0, 1, len(predicted_voltage_trajectories)))
        time_shift = 288 

        for idx, (predicted, actual) in enumerate(zip(predicted_voltage_trajectories, actual_voltage_trajectories)):
            if idx != 0:
                continue

            pred_time, pred_voltage = np.array(predicted["time"]), np.array(predicted["voltage"])
            actual_time, actual_voltage = np.array(actual["time"]), np.array(actual["voltage"])

            # Interpolate predicted voltage to match actual time points
            pred_voltage_interp = np.interp(actual_time, pred_time, pred_voltage)

            # Compute RMSE and MAE
            rmse = np.sqrt(np.mean((pred_voltage_interp - actual_voltage) ** 2))
            mae = np.mean(np.abs(pred_voltage_interp - actual_voltage))

            print(f"Voltage RMSE: {rmse:.4f} V")
            print(f"Voltage MAE: {mae:.4f} V")

            ax.plot(pred_time + time_shift, pred_voltage-2.2-3.8, linestyle="dashed", color=colors[idx], linewidth=1.2, alpha=0.8, label="Predicted Voltage")
            ax.plot(actual_time + time_shift, actual_voltage-2.2-3.8, linestyle="solid", color=colors[idx], linewidth=1.2, alpha=0.8, label="Actual Voltage")

        # Compute Mid-Time for Vertical Line
        mid_time = (actual_time[0] + actual_time[-1]) / 2

        # Add Voltage Threshold Line at 18V (Red Dashed Line)
        ax.axhline(y=18, color='red', linestyle='dashed', linewidth=1.5, label="Voltage Threshold")

        # Add Mid-Time Indicator (Blue Dashed Line)
        # ax.axvline(x=mid_time, color='blue', linestyle='dashed', linewidth=1.5, label="Mid-Flight Incident")

        # # Shade the Right Half of the Plot
        # ax.axvspan(mid_time, actual_time[-1], color='gray', alpha=0.3)

        ax.set_xlabel("Time (s)", fontsize=12)
        ax.set_ylabel("Voltage (V)", fontsize=12)
        ax.grid(True, linestyle="--", alpha=0.7)

        # **Compact Legend Inside the Plot**
        ax.legend(loc="lower left", fontsize=10, frameon=False)

        plt.savefig(BASE_DIR / "results" / "Voltage_Comparison.pdf", format="pdf", bbox_inches="tight")
        plt.show()
        plt.close()



if __name__ == "__main__":
    BASE_DIR = Path(__file__).resolve().parents[1]
    MODEL_PATH = BASE_DIR / "saved_models" / "neural_ode_epoch_new.pth"
    CURRENT_PROFILE_PATH = BASE_DIR / "data" / "SimulationResults" /"new_sim_data"/  "saved_current_profiles_short.pkl"
    TRAIN_PATH = BASE_DIR / "data" / "training_data.pkl"
    TEST_PATH = BASE_DIR / "data" / "test_data.pkl"
    predictor = VoltagePredictor(MODEL_PATH, CURRENT_PROFILE_PATH, TRAIN_PATH, TEST_PATH)
    predicted_voltage_trajectories = predictor.predict_voltage_new()
    actual_voltage_trajectories = predictor.simulate_actual_battery(predicted_voltage_trajectories)
    predicted_voltage_trajectories = predictor.load_saved_current_profiles(BASE_DIR/ "new_voltages" /"saved_predicted_voltage_trajectories.pkl")
    actual_voltage_trajectories = predictor.load_saved_current_profiles(BASE_DIR/ "new_voltages" /"saved_actual_voltage_trajectories.pkl")
    predictor.clipped_current_profiles = predictor.load_saved_current_profiles(BASE_DIR/ "new_voltages"/ "saved_clipped_current_profiles.pkl")

    predictor.plot_comparison_voltage_trajectories(predicted_voltage_trajectories, actual_voltage_trajectories)