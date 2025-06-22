import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import scipy.interpolate as interp
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm 
import sys
import os 
import time 
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# Configure Matplotlib
plt.rcParams["text.usetex"] = True
from scripts.generate_dataset import TarotBattery

class BatteryDataset(Dataset):
    def __init__(self, current_profiles, initial_voltages):
        self.currents = [torch.tensor(profile['current'], dtype=torch.float32) for profile in current_profiles]
        self.times = [torch.tensor(profile['time'], dtype=torch.float32) for profile in current_profiles]
        self.initial_voltages = torch.tensor(initial_voltages, dtype=torch.float32)

    def __len__(self):
        return len(self.currents)

    def __getitem__(self, idx):
        return self.currents[idx], self.times[idx], self.initial_voltages[idx]

class LSTMPINNBatteryModel(nn.Module):
    def __init__(self, hidden_size, output_size, num_layers=1):
        super(LSTMPINNBatteryModel, self).__init__()
        self.lstm = nn.LSTM(input_size=3, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, current, times, initial_voltage):
        if current.dim() == 2:
            current = current.unsqueeze(-1)
        if times.dim() == 2:
            times = times.unsqueeze(-1)

        batch_size, seq_len = current.size(0), current.size(1)
        if initial_voltage.dim() == 1:
            initial_voltage = initial_voltage.unsqueeze(-1)
        initial_voltage_expanded = initial_voltage.unsqueeze(1).expand(batch_size, seq_len, 1)

        x = torch.cat([current, times, initial_voltage_expanded], dim=2)
        h0 = torch.zeros(self.lstm.num_layers, batch_size, self.lstm.hidden_size).to(x.device)
        c0 = torch.zeros(self.lstm.num_layers, batch_size, self.lstm.hidden_size).to(x.device)

        lstm_out, _ = self.lstm(x, (h0, c0))
        output = self.fc(lstm_out)
        return output

def timestep_standardize(data, times, full_data=None, full_times=None):
    data_to_standardize = full_data if full_data is not None else data
    times_to_standardize = full_times if full_times is not None else times

    max_len = max(len(seq) for seq in data_to_standardize)
    full_time_grid = next(t for t in times_to_standardize if len(t) == max_len)

    data_matrix = np.zeros((len(data_to_standardize), max_len))
    mask = np.zeros((len(data_to_standardize), max_len), dtype=bool)

    for i, (seq, seq_times) in enumerate(zip(data_to_standardize, times_to_standardize)):
        indices = np.searchsorted(full_time_grid, seq_times, side='left')
        indices = np.clip(indices, 0, max_len - 1)
        time_diffs = np.abs(full_time_grid[indices] - seq_times)
        valid = time_diffs < 1e-6
        data_matrix[i, indices[valid]] = seq[valid]
        mask[i, indices[valid]] = True

    mean = np.zeros(max_len)
    std = np.ones(max_len)
    for t_idx in range(max_len):
        values = data_matrix[:, t_idx][mask[:, t_idx]]
        if len(values) > 0:
            mean[t_idx] = np.mean(values)
            std[t_idx] = np.std(values) if np.std(values) > 0 else 1
    std[std < 1e-6] = 1

    standardized_data = []
    for seq, seq_times in zip(data, times):
        indices = np.searchsorted(full_time_grid, seq_times, side='left')
        indices = np.clip(indices, 0, max_len - 1)
        seq_mean = mean[indices]
        seq_std = std[indices]
        standardized_seq = (seq - seq_mean) / seq_std
        standardized_data.append(standardized_seq)

    return standardized_data, mean, std, full_time_grid

def standardize(data, times, mean, std, time_grid):
    if np.isscalar(data) or (isinstance(data, np.ndarray) and data.size == 1):
        data = np.array([data]) if np.isscalar(data) else data.flatten()
        times = np.array([times]) if np.isscalar(times) else times.flatten()
        idx = np.searchsorted(time_grid, times[0], side='left')
        idx = np.clip(idx, 0, len(time_grid) - 1)
        seq_mean = mean[idx]
        seq_std = std[idx]
        result = (data - seq_mean) / (seq_std + 1e-6)
        return result[0]
    else:
        indices = np.searchsorted(time_grid, times, side='left')
        indices = np.clip(indices, 0, len(time_grid) - 1)
        seq_mean = mean[indices]
        seq_std = std[indices]
        return (data - seq_mean) / (seq_std + 1e-6)

def destandardize(data, times, mean, std, time_grid):
    if isinstance(data, torch.Tensor):
        mean = torch.tensor(mean, dtype=torch.float32, device=data.device) if not isinstance(mean, torch.Tensor) else mean.to(data.device)
        std = torch.tensor(std, dtype=torch.float32, device=data.device) if not isinstance(std, torch.Tensor) else std.to(data.device)
        time_grid = torch.tensor(time_grid, dtype=torch.float32, device=data.device) if not isinstance(time_grid, torch.Tensor) else time_grid.to(data.device)
        times = times.to(data.device)

        if data.dim() == 0 or (data.dim() == 1 and data.numel() == 1):
            data = data.view(1, 1)
            times = times.view(1, 1)
            indices = torch.searchsorted(time_grid, times[:, 0], right=False)
            indices = torch.clamp(indices, 0, len(time_grid) - 1)
            seq_mean = mean[indices]
            seq_std = std[indices]
            result = data * (seq_std + 1e-6) + seq_mean
            return result.squeeze()
        else:
            if data.dim() == 1:
                data = data.unsqueeze(0)
                times = times.unsqueeze(0)
            time_diffs = torch.abs(times.unsqueeze(-1) - time_grid)
            indices = torch.argmin(time_diffs, dim=-1)
            seq_mean = mean[indices]
            seq_std = std[indices]
            mask = (data != 0).float()
            result = data * (seq_std + 1e-6) + seq_mean
            result = result * mask
            if data.shape[0] == 1:
                result = result.squeeze(0)
            return result
    else:
        if np.isscalar(data) or (isinstance(data, np.ndarray) and data.size == 1):
            data = np.array([data]) if np.isscalar(data) else data.flatten()
            times = np.array([times]) if np.isscalar(times) else times.flatten()
            idx = np.searchsorted(time_grid, times[0], side='left')
            idx = np.clip(idx, 0, len(time_grid) - 1)
            seq_mean = mean[idx]
            seq_std = std[idx]
            result = data * (seq_std + 1e-6) + seq_mean
            return result[0]
        else:
            indices = np.searchsorted(time_grid, times, side='left')
            indices = np.clip(indices, 0, len(time_grid) - 1)
            seq_mean = mean[indices]
            seq_std = std[indices]
            return data * (seq_std + 1e-6) + seq_mean

def load_data(training_file, test_file):
    with open(training_file, "rb") as f:
        training_data = pickle.load(f)
    with open(test_file, "rb") as f:
        test_data = pickle.load(f)

    train_inputs = [np.array(d['input']) for d in training_data]
    train_outputs = [np.array(d['output']) for d in training_data]
    train_times = [np.array(d['time']) for d in training_data]
    test_inputs = [np.array(d['input']) for d in test_data]
    test_outputs = [np.array(d['output']) for d in test_data]
    test_times = [np.array(d['time']) for d in test_data]

    max_len = max(len(seq) for seq in train_inputs)
    full_train_inputs = [seq for seq in train_inputs if len(seq) == max_len]
    full_train_times = [t for t in train_times if len(t) == max_len]
    full_train_outputs = [seq for seq in train_outputs if len(seq) == max_len]

    _, input_mean, input_std, time_grid = timestep_standardize(
        train_inputs, train_times, full_data=full_train_inputs, full_times=full_train_times
    )
    _, output_mean, output_std, _ = timestep_standardize(
        train_outputs, train_times, full_data=full_train_outputs, full_times=full_train_times
    )

    return (input_mean, input_std, time_grid), (output_mean, output_std)

class VoltagePredictor:
    def __init__(self, model_path, file_path, training_file, test_file):
        self.model_path = model_path
        self.current_profiles = self.load_saved_current_profiles(file_path)
        self.current_range, self.voltage_range = load_data(training_file, test_file)
        self.current_mean, self.current_std, self.time_grid = self.current_range
        self.voltage_mean, self.voltage_std = self.voltage_range
        self.model = LSTMPINNBatteryModel(hidden_size=128, output_size=1)
        self.model.load_state_dict(torch.load(model_path))
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()
        self.clipped_current_profiles = None 
        self.battery_simulator = TarotBattery(time_step=0.1)

        # Convert standardization stats to tensors
        self.current_mean = torch.tensor(self.current_mean, dtype=torch.float32, device=self.device)
        self.current_std = torch.tensor(self.current_std, dtype=torch.float32, device=self.device)
        self.voltage_mean = torch.tensor(self.voltage_mean, dtype=torch.float32, device=self.device)
        self.voltage_std = torch.tensor(self.voltage_std, dtype=torch.float32, device=self.device)
        self.time_grid = torch.tensor(self.time_grid, dtype=torch.float32, device=self.device)

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
        
        # Initial voltages for each profile (standardized)
        initial_voltages = []
        x0 = torch.tensor(27.4, dtype=torch.float32, device=self.device)  # Initial voltage
        # Standardize the initial voltage
        initial_time = self.current_profiles[0]['time'][0]
        x0_standardized = standardize(x0.cpu().numpy(), initial_time, 
                                    self.current_mean.cpu().numpy(), 
                                    self.current_std.cpu().numpy(), 
                                    self.time_grid.cpu().numpy())
        x0_standardized = torch.tensor(x0_standardized, dtype=torch.float32, device=self.device)
        
        for idx in range(len(self.current_profiles)):
            initial_voltages.append(x0_standardized.item())
        
        current_inputs = BatteryDataset(self.current_profiles, initial_voltages)
        current_loader = DataLoader(current_inputs, batch_size=1, shuffle=False)

        for idx, (current_profile, time_horizon, init_vol) in enumerate(current_loader):
            start_time = time.time()

            # Move tensors to device
            current_profile = current_profile.to(self.device)
            time_horizon = time_horizon.to(self.device)
            init_vol = init_vol.to(self.device)

            # Clip current profiles
            current_values = current_profile[:, :max_length]
            times = time_horizon[:, :max_length]
            self.clipped_current_profiles.append({
                "current": current_values[0].cpu().numpy(), 
                "time": times[0].cpu().numpy()
            })

            # Standardize current values
            standardized_current = torch.tensor(
                standardize(current_values.cpu().numpy(), times.cpu().numpy(), 
                           self.current_mean.cpu().numpy(), self.current_std.cpu().numpy(), 
                           self.time_grid.cpu().numpy()),
                dtype=torch.float32, device=self.device
            )

            with torch.no_grad():
                output = self.model(standardized_current, times, init_vol)
                output = output.squeeze(-1)
                
                # Destandardize the predicted voltage
                voltage = destandardize(output, times, self.voltage_mean, self.voltage_std, self.time_grid)
                voltage = voltage.cpu().numpy()
                # print(voltage.shape)
                # print(times.shape)
                # plt.plot(times[0].cpu().numpy(), voltage)
                # plt.show()

            # Store voltage trajectory
            predicted_voltage_trajectories.append({
                "time": times[0].cpu().numpy(), 
                "voltage": voltage
            })

            # Prepare initial voltage for the next prediction
            if idx < len(self.current_profiles) - 1:
                next_initial_time = self.current_profiles[idx + 1]['time'][0]
                time_index = np.searchsorted(times[0].cpu().numpy(), next_initial_time)
                if time_index >= len(output[0]):
                    time_index = len(output[0]) - 1
                initial_voltage_standardized = output[0, time_index]
                x0_standardized = initial_voltage_standardized.unsqueeze(0)
                initial_voltages[idx + 1] = x0_standardized.item()

            print(f"Prediction {idx}: Time taken = {time.time() - start_time:.2f} seconds")

        return predicted_voltage_trajectories

    def simulate_actual_battery(self, predicted_voltage_trajectories):
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
                actual_voltage_trajectories.append({
                    "time": time.tolist(), 
                    "voltage": voltage_trajectory.tolist()
                })

        return actual_voltage_trajectories

    def plot_comparison_voltage_trajectories(self, predicted_voltage_trajectories, actual_voltage_trajectories):
        fig, ax = plt.subplots(figsize=(5, 4), dpi=300)
        colors = plt.cm.viridis(np.linspace(0, 1, len(predicted_voltage_trajectories)))
        time_shift =0# 288 
        batt_calibration = 0 #-2.2-3.8

        for idx, (predicted, actual) in enumerate(zip(predicted_voltage_trajectories, actual_voltage_trajectories)):
            if idx != 50:
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

            ax.plot(pred_time + time_shift, pred_voltage - batt_calibration, linestyle="dashed", color=colors[idx], linewidth=1.2, alpha=0.8, label="Predicted Voltage")
            ax.plot(actual_time + time_shift, actual_voltage - batt_calibration, linestyle="solid", color=colors[idx], linewidth=1.2, alpha=0.8, label="Actual Voltage")

        # Compute Mid-Time for Vertical Line
        mid_time = (actual_time[0] + actual_time[-1]) / 2

        # Add Voltage Threshold Line at 18V
        ax.axhline(y=18, color='red', linestyle='dashed', linewidth=1.5, label="Voltage Threshold")

        ax.set_xlabel("Time (s)", fontsize=12)
        ax.set_ylabel("Voltage (V)", fontsize=12)
        ax.grid(True, linestyle="--", alpha=0.7)

        # ax.legend(loc="lower left", fontsize=10, frameon=False)

        plt.savefig(BASE_DIR / "results" / "Voltage_Comparison.pdf", format="pdf", bbox_inches="tight")
        plt.show()
        plt.close()

if __name__ == "__main__":
    BASE_DIR = Path(__file__).resolve().parents[1]
    TRAIN_PATH = BASE_DIR /"data" /"training_data_mixed_70.pkl"
    TEST_PATH = BASE_DIR /"data"/ "test_data_mixed_30.pkl"
    MODEL_PATH = BASE_DIR /"saved_models"/ 'trained_pinn_model_actual.pth'
    CURRENT_PROFILE_PATH = BASE_DIR /"data" /"SimulationResults" / "saved_current_profiles_long.pkl"
    predictor = VoltagePredictor(MODEL_PATH, CURRENT_PROFILE_PATH, TRAIN_PATH, TEST_PATH)
    predicted_voltage_trajectories = predictor.predict_voltage()
    actual_voltage_trajectories = predictor.simulate_actual_battery(predicted_voltage_trajectories)
    predictor.plot_comparison_voltage_trajectories(predicted_voltage_trajectories, actual_voltage_trajectories)