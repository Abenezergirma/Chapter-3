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
import sys
import time 
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
plt.rcParams["text.usetex"] = True
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
    """ Standardize the data using dynamically selected or extrapolated mean and std segments. """
    # Determine the indices for the mean/std based on the time entries
    indices = np.interp(times, training_time, np.arange(len(training_time)))
    rounded_indices = np.round(indices).astype(int)
    # Ensure that indices do not exceed the length of full_mean and full_std
    max_index = len(full_mean) - 1
    rounded_indices[rounded_indices > max_index] = max_index

    if max(rounded_indices) < len(data) - 1:
        # If indices for standardization are less than the data length, extrapolate
        extended_mean = extrapolate_array(full_mean, len(data))
        extended_std = extrapolate_array(full_std, len(data))
        return (data - extended_mean) / extended_std
    else:
        # Use the indices directly if within bounds
        selected_mean = full_mean[rounded_indices]
        selected_std = full_std[rounded_indices]
        return (data - selected_mean) / selected_std

def destandardize(data, times, full_mean, full_std, training_time):
    """ Destandardize the data using dynamically selected or extrapolated mean and std segments. """
    indices = np.interp(times, training_time, np.arange(len(training_time)))
    rounded_indices = np.round(indices).astype(int)
    max_index = len(full_mean) - 1
    rounded_indices[rounded_indices > max_index] = max_index

    if max(rounded_indices) < len(data) - 1:
        extended_mean = extrapolate_array(full_mean, len(data))
        extended_std = extrapolate_array(full_std, len(data))
        return data * extended_std + extended_mean
    else:
        selected_mean = full_mean[rounded_indices]
        selected_std = full_std[rounded_indices]
        return data * selected_std + selected_mean

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
        # Visualize standardization
    # fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    # axs[0, 0].plot(input_mean, label='Mean of Inputs')
    # axs[0, 0].plot(input_std, label='Std Dev of Inputs')
    # axs[0, 0].set_title('Input Statistics')
    # axs[0, 0].legend()

    # axs[0, 1].plot(output_mean, label='Mean of Outputs')
    # axs[0, 1].plot(output_std, label='Std Dev of Outputs')
    # axs[0, 1].set_title('Output Statistics')
    # axs[0, 1].legend()
    # plt.tight_layout()
    # plt.show()

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
    


    # def predict_voltage(self):
    #     predicted_voltage_trajectories = []
    #     max_length = len(self.current_profiles[0]["current"])  # Length of the first current trajectory
    #     self.clipped_current_profiles = []  # Storing clipped current profiles

    #     # Initial voltage for the first prediction
    #     initial_voltage_standardized = 0.0
    #     x0 = torch.tensor([[initial_voltage_standardized]], dtype=torch.float32)

    #     for idx, current_profile in enumerate(self.current_profiles):
            
    #         start_time = time.time()

    #         # Clipping current profiles
    #         current_values = current_profile["current"][:max_length]
    #         times = current_profile["time"][:max_length]
    #         self.clipped_current_profiles.append({"current": current_values, "time": times})
    #         # plt.plot(self.voltage_mean)
    #         # plt.show()

    #         standardized_current = standardize(current_values, times, self.current_mean, self.current_std, self.training_time)
    #         current_tensor = torch.tensor(standardized_current, dtype=torch.float32).unsqueeze(0).unsqueeze(-1)

    #         t = torch.linspace(0, 1, len(current_values))  # Assuming uniform time steps
    #         self.ode_func.set_current_profile(current_tensor)
    #         self.ode_func.set_time_tensor(t)
    #         pred_output = self.neural_ode(x0, current_tensor, t).squeeze(0).squeeze(-1)
    #         voltage = destandardize(pred_output.detach().numpy(), times, self.voltage_mean, self.voltage_std, self.training_time)

    #         # Store voltage trajectory for visualization or further use
    #         predicted_voltage_trajectories.append({"time": times, "voltage": voltage})

    #         # Print and plot results
    #         print(f"Prediction {idx}: Time taken = {time.time() - start_time:.2f} seconds")
    #         plt.plot(times, voltage)
    #         plt.show()

    #         if idx < len(self.current_profiles) - 1:
    #             # Prepare initial voltage for the next prediction
    #             next_initial_time = self.current_profiles[idx + 1]['time'][0]
    #             time_index = np.searchsorted(times, next_initial_time)
    #             initial_voltage_standardized = pred_output[time_index] #- self.voltage_mean[0]) / self.voltage_std[0]
    #             x0 = torch.tensor([[initial_voltage_standardized]], dtype=torch.float32)

    #     with open("saved_predicted_voltage_trajectories.pkl", "wb") as f:
    #         pickle.dump(predicted_voltage_trajectories, f)
    #     print("predicted_voltage_trajectories saved to 'predicted_voltage_trajectories.pkl'.")

    #     with open("saved_clipped_current_profiles.pkl", "wb") as f:
    #         pickle.dump(self.clipped_current_profiles, f)
    #     print("clipped_current_profiles saved to 'clipped_current_profiles.pkl'.")   
    #     return predicted_voltage_trajectories

    def predict_voltage(self):
        predicted_voltage_trajectories = []
        max_length = len(self.current_profiles[0]["current"])  # Length of the first current trajectory
        self.clipped_current_profiles = []  # Storing clipped current profiles

        for idx, current_profile in enumerate(self.current_profiles):
            start_time = time.time()

            # Clipping current profiles
            current_values = current_profile["current"][:max_length]
            times = current_profile["time"][:max_length]
            self.clipped_current_profiles.append({"current": current_values, "time": times})

            # Standardize current using actual times
            standardized_current = standardize(current_values, times, self.current_mean, self.current_std, self.training_time)
            current_tensor = torch.tensor(standardized_current, dtype=torch.float32).unsqueeze(0).unsqueeze(-1)

            # Use actual time range for ODE integration
            t = torch.tensor(times / times[-1], dtype=torch.float32)  # Normalize to [0, 1] based on max time
            if idx == 0 and times[0] != 0:
                # Simulate from time 0 to times[0] to get initial voltage
                ref_times = np.linspace(0, times[0], max_length)
                # Assume a reference current profile (e.g., zero current or from training data)
                ref_current = self.current_mean[0:max_length]  # Placeholder: replace with actual reference current
                ref_std_current = standardize(ref_current, ref_times, self.current_mean, self.current_std, self.training_time)
                ref_current_tensor = torch.tensor(ref_std_current, dtype=torch.float32).unsqueeze(0).unsqueeze(-1)
                ref_t = torch.tensor(ref_times / ref_times[-1], dtype=torch.float32)
                x0 = torch.tensor([[0.0]], dtype=torch.float32)  # Standardized initial voltage at t=0
                self.ode_func.set_current_profile(ref_current_tensor)
                self.ode_func.set_time_tensor(ref_t)
                ref_output = self.neural_ode(x0, ref_current_tensor, ref_t).squeeze(0).squeeze(-1)
                initial_voltage_standardized = ref_output[-1].item()  # Voltage at times[0]
                x0 = torch.tensor([[initial_voltage_standardized]], dtype=torch.float32)
            else:
                # Use standard initial voltage for time[0] = 0 or subsequent profiles
                initial_voltage_standardized = 0.0 if idx == 0 else pred_output[time_index]
                x0 = torch.tensor([[initial_voltage_standardized]], dtype=torch.float32)

            # Predict voltage for current profile
            self.ode_func.set_current_profile(current_tensor)
            self.ode_func.set_time_tensor(t)
            pred_output = self.neural_ode(x0, current_tensor, t).squeeze(0).squeeze(-1)
            voltage = destandardize(pred_output.detach().numpy(), times, self.voltage_mean, self.voltage_std, self.training_time)

            # Store voltage trajectory
            predicted_voltage_trajectories.append({"time": times, "voltage": voltage})

            # Print and plot results
            print(f"Prediction {idx}: Time taken = {time.time() - start_time:.2f} seconds")
            # plt.plot(times, voltage)
            # plt.xlabel("Time (s)")
            # plt.ylabel("Voltage (V)")
            # plt.grid(True)
            # plt.show()

            # Prepare initial voltage for the next prediction
            if idx < len(self.current_profiles) - 1:
                next_initial_time = self.current_profiles[idx + 1]['time'][0]
                time_index = np.searchsorted(times, next_initial_time)
                if time_index < len(pred_output):
                    initial_voltage_standardized = pred_output[time_index].item()
                else:
                    print(f"Warning: next_initial_time {next_initial_time} out of bounds for profile {idx}")
                    initial_voltage_standardized = pred_output[-1].item()  # Fallback to last voltage

        # Save results
        with open("saved_predicted_voltage_trajectories.pkl", "wb") as f:
            pickle.dump(predicted_voltage_trajectories, f)
        print("predicted_voltage_trajectories saved to 'predicted_voltage_trajectories.pkl'.")

        with open("saved_clipped_current_profiles.pkl", "wb") as f:
            pickle.dump(self.clipped_current_profiles, f)
        print("clipped_current_profiles saved to 'saved_clipped_current_profiles.pkl'.")
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

            self.battery_simulator.set_initial_voltage(predicted_initial_voltage, current)
            voltage_trajectory = self.battery_simulator.simulate_battery(current, predicted_initial_voltage)

            if voltage_trajectory is not None:
                actual_voltage_trajectories.append({"time": time.tolist(), "voltage": voltage_trajectory.tolist()})

        with open("saved_actual_voltage_trajectories.pkl", "wb") as f:
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
            pred_time, pred_voltage = np.array(predicted["time"]), np.array(predicted["voltage"])
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
            ax.plot(actual_time + time_shift, actual_voltage - batt_calibration, 
                    linestyle="solid", color=colors[idx], linewidth=0.8, alpha=0.7)

        # Plot the first trajectory with labels for legend
        if num_trajectories > 0:
            first_pred = predicted_voltage_trajectories[0]
            first_actual = actual_voltage_trajectories[0]
            ax.plot((first_pred["time"] + time_shift), (np.array(first_pred["voltage"]) - batt_calibration), 
                    linestyle="dashed", color=colors[0], linewidth=0.8, alpha=0.7, label="Predicted Voltage")
            
            ax.plot(np.array(first_actual["time"]) , (np.array(first_actual["voltage"]) - batt_calibration), 
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


    # def plot_comparison_voltage_trajectories(self, predicted_voltage_trajectories, actual_voltage_trajectories):
    #     fig, ax = plt.subplots(figsize=(5, 4), dpi=300)
    #     num_trajectories = min(50, len(predicted_voltage_trajectories))
    #     colors = plt.cm.viridis(np.linspace(0, 1, num_trajectories))
    #     time_shift = 0
    #     batt_calibration = 0

    #     for idx, (predicted, actual) in enumerate(zip(predicted_voltage_trajectories[:50], actual_voltage_trajectories[:50])):
    #         if idx != 0:
    #             continue
    #         pred_time, pred_voltage = np.array(predicted["time"]), np.array(predicted["voltage"])
    #         actual_time, actual_voltage = np.array(actual["time"]), np.array(actual["voltage"])

    #         # Validate arrays
    #         if pred_time.size == 0 or pred_voltage.size == 0 or actual_time.size == 0 or actual_voltage.size == 0:
    #             print(f"Skipping trajectory {idx+1} due to empty arrays.")
    #             continue
    #         if pred_time.shape != pred_voltage.shape or actual_time.shape != actual_voltage.shape:
    #             print(f"Skipping trajectory {idx+1} due to shape mismatch: "
    #                 f"pred_time shape={pred_time.shape}, pred_voltage shape={pred_voltage.shape}, "
    #                 f"actual_time shape={actual_time.shape}, actual_voltage shape={actual_voltage.shape}")
    #             continue

    #         # Interpolate predicted voltage to match actual time points
    #         try:
    #             pred_voltage_interp = np.interp(actual_time, pred_time, pred_voltage)
    #         except ValueError as e:
    #             print(f"Skipping trajectory {idx+1} due to interpolation error: {e}")
    #             continue

    #         # Compute RMSE and MAE
    #         rmse = np.sqrt(np.mean((pred_voltage_interp - actual_voltage) ** 2))
    #         mae = np.mean(np.abs(pred_voltage_interp - actual_voltage))
    #         print(f"Trajectory {idx+1} Voltage RMSE: {rmse:.4f} V")
    #         print(f"Trajectory {idx+1} Voltage MAE: {mae:.4f} V")

    #         # Plot trajectories
    #         ax.plot(pred_time + time_shift, pred_voltage - batt_calibration, 
    #                 linestyle="dashed", color=colors[idx], linewidth=0.8, alpha=0.7)
    #         ax.plot(actual_time + time_shift, actual_voltage - batt_calibration, 
    #                 linestyle="solid", color=colors[idx], linewidth=0.8, alpha=0.7)

    #     # Plot the first trajectory with labels for legend
    #     if num_trajectories > 0:
    #         first_pred = predicted_voltage_trajectories[0]
    #         first_actual = actual_voltage_trajectories[0]
    #         ax.plot(first_pred["time"] + time_shift, np.array(first_pred["voltage"]) - batt_calibration, 
    #                 linestyle="dashed", color=colors[0], linewidth=0.8, alpha=0.7, label="Predicted Voltage")
    #         ax.plot(first_actual["time"] , np.array(first_actual["voltage"]  ), 
    #                 linestyle="solid", color=colors[0], linewidth=0.8, alpha=0.7, label="Actual Voltage")

    #     # Add voltage threshold line at 18V
    #     ax.axhline(y=20.5, color='red', linestyle='dashed', linewidth=1.5, label="Voltage Threshold")

    #     # Set labels and grid
    #     ax.set_xlabel("Time (s)", fontsize=12)
    #     ax.set_ylabel("Voltage (V)", fontsize=12)
    #     ax.grid(True, linestyle="--", alpha=0.7)

    #     # Add colorbar for trajectory index
    #     if num_trajectories > 0:
    #         sm = ScalarMappable(cmap=plt.cm.viridis, norm=Normalize(vmin=1, vmax=num_trajectories))
    #         cbar = fig.colorbar(sm, ax=ax, aspect=30, pad=0.02)
    #         cbar.set_label("Trajectory Index", fontsize=10)

    #     # Add legend
    #     ax.legend(loc="best", fontsize=10, frameon=False)

    #     # Save and show plot
    #     plt.savefig(BASE_DIR / "results" / "Voltage_Comparison.pdf", format="pdf", bbox_inches="tight")
    #     plt.show()
    #     plt.close()

if __name__ == "__main__":
    BASE_DIR = Path(__file__).resolve().parents[1]
    MODEL_PATH = BASE_DIR / "saved_models" / "neural_ode_epoch_new.pth"
    CURRENT_PROFILE_PATH = BASE_DIR / "data" / "SimulationResults" / "saved_current_profiles_long.pkl"
    TRAIN_PATH = BASE_DIR / "data" / "training_data.pkl"
    TEST_PATH = BASE_DIR / "data" / "test_data.pkl"
    predictor = VoltagePredictor(MODEL_PATH, CURRENT_PROFILE_PATH, TRAIN_PATH, TEST_PATH)
    # predicted_voltage_trajectories = predictor.predict_voltage()
    # actual_voltage_trajectories = predictor.simulate_actual_battery(predicted_voltage_trajectories)
    predicted_voltage_trajectories = predictor.load_saved_current_profiles(BASE_DIR/ "voltage NODE" /"saved_predicted_voltage_trajectories.pkl")
    actual_voltage_trajectories = predictor.load_saved_current_profiles(BASE_DIR / "voltage NODE" /"saved_actual_voltage_trajectories.pkl")
    predictor.clipped_current_profiles = predictor.load_saved_current_profiles(BASE_DIR / "voltage NODE" / "saved_clipped_current_profiles.pkl")
    predictor.plot_comparison_voltage_trajectories(predicted_voltage_trajectories, actual_voltage_trajectories)