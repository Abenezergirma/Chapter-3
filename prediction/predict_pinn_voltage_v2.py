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
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
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


def destandardize(data, times, mean, std, time_grid):
    # Convert to tensors if needed, ensure float32
    is_tensor = isinstance(data, torch.Tensor)
    device = data.device if is_tensor else None
    for x in [data, times, mean, std, time_grid]:
        if is_tensor and not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32, device=device)
        elif is_tensor:
            x = x.to(dtype=torch.float32, device=device)
        else:
            x = np.asarray(x)

    # Validate time_grid
    time_grid_np = time_grid.cpu().numpy() if is_tensor else time_grid
    if len(time_grid_np) < 2 or not np.all(np.diff(time_grid_np) >= 0):
        raise ValueError("time_grid must be a sorted array with at least two points")

    # Handle scalar case
    if (is_tensor and (data.dim() == 0 or data.numel() == 1)) or (not is_tensor and data.size == 1):
        data = data.view(1) if is_tensor else data.flatten()
        times = times.view(1) if is_tensor else times.flatten()
        t = times[0] if is_tensor else times[0]
        if t <= time_grid[0]:
            seq_mean, seq_std = mean[0], std[0]
        elif t >= time_grid[-1]:
            t0, t1 = time_grid[-2], time_grid[-1]
            m0, m1 = mean[-2], mean[-1]
            s0, s1 = std[-2], std[-1]
            seq_mean = m0 + (m1 - m0) * (t - t0) / (t1 - t0)
            seq_std = s0 + (s1 - s0) * (t - t0) / (t1 - t0)
        else:
            seq_mean = torch.tensor(np.interp(t, time_grid_np, mean.cpu().numpy()), dtype=torch.float32, device=device) if is_tensor else np.interp(t, time_grid, mean)
            seq_std = torch.tensor(np.interp(t, time_grid_np, std.cpu().numpy()), dtype=torch.float32, device=device) if is_tensor else np.interp(t, time_grid, std)
        result = data * (seq_std + 1e-6) + seq_mean
        return result.squeeze() if is_tensor else result[0]

    # Array case
    if is_tensor and data.dim() == 1:
        data, times = data.unsqueeze(0), times.unsqueeze(0)
    
    # Initialize outputs
    seq_mean = torch.zeros_like(times, dtype=torch.float32, device=device) if is_tensor else np.zeros_like(times)
    seq_std = torch.zeros_like(times, dtype=torch.float32, device=device) if is_tensor else np.zeros_like(times)
    
    # Masks
    before_mask = times <= time_grid[0]
    after_mask = times >= time_grid[-1]
    within_mask = ~(before_mask | after_mask)
    
    # Before time_grid[0]
    seq_mean[before_mask] = mean[0]
    seq_std[before_mask] = std[0]
    
    # Within time_grid
    if within_mask.any():
        times_within = times[within_mask].cpu().numpy() if is_tensor else times[within_mask]
        seq_mean[within_mask] = torch.tensor(np.interp(times_within, time_grid_np, mean.cpu().numpy()), dtype=torch.float32, device=device) if is_tensor else np.interp(times_within, time_grid, mean)
        seq_std[within_mask] = torch.tensor(np.interp(times_within, time_grid_np, std.cpu().numpy()), dtype=torch.float32, device=device) if is_tensor else np.interp(times_within, time_grid, std)
    
    # After time_grid[-1]
    if after_mask.any():
        t0, t1 = time_grid[-2], time_grid[-1]
        m0, m1 = mean[-2], mean[-1]
        s0, s1 = std[-2], std[-1]
        slope_mean = (m1 - m0) / (t1 - t0)
        slope_std = (s1 - s0) / (t1 - t0)
        seq_mean[after_mask] = m0 + slope_mean * (times[after_mask] - t0)
        seq_std[after_mask] = s0 + slope_std * (times[after_mask] - t0)
    
    result = data * (seq_std + 1e-6) + seq_mean
    if is_tensor and data.shape[0] == 1:
        result = result.squeeze(0)
    return result
        
        
def standardize(data, times, mean, std, time_grid):
    # Validate time_grid
    if not isinstance(time_grid, (np.ndarray, torch.Tensor)) or len(time_grid) == 0:
        raise ValueError("time_grid must be a non-empty array or tensor")
    if not np.all(np.diff(time_grid) >= 0):
        raise ValueError("time_grid must be sorted")

    # Convert inputs to NumPy for consistency
    data = np.asarray(data)
    times = np.asarray(times)
    mean = np.asarray(mean)
    std = np.asarray(std)
    time_grid = np.asarray(time_grid)

    if data.size == 1:
        # Scalar case
        data = data.flatten()
        times = times.flatten()
        # Interpolate mean and std at times[0]
        seq_mean = np.interp(times[0], time_grid, mean)
        seq_std = np.interp(times[0], time_grid, std)
        result = (data - seq_mean) / (seq_std + 1e-6)
        return result[0]
    else:
        # Array case
        # Interpolate mean and std for all times
        seq_mean = np.interp(times, time_grid, mean)
        seq_std = np.interp(times, time_grid, std)
        return (data - seq_mean) / (seq_std + 1e-6)

def destandardize_old(data, times, mean, std, time_grid):
    # Validate time_grid
    if not isinstance(time_grid, (np.ndarray, torch.Tensor)) or len(time_grid) == 0:
        raise ValueError("time_grid must be a non-empty array or tensor")
    if not np.all(np.diff(time_grid) >= 0):
        raise ValueError("time_grid must be sorted")

    if isinstance(data, torch.Tensor):
        # Convert inputs to tensors on data's device
        mean = torch.tensor(mean, dtype=torch.float32, device=data.device) if not isinstance(mean, torch.Tensor) else mean.to(data.device)
        std = torch.tensor(std, dtype=torch.float32, device=data.device) if not isinstance(std, torch.Tensor) else std.to(data.device)
        time_grid = torch.tensor(time_grid, dtype=torch.float32, device=data.device) if not isinstance(time_grid, torch.Tensor) else time_grid.to(data.device)
        times = torch.tensor(times, dtype=torch.float32, device=data.device) if not isinstance(times, torch.Tensor) else times.to(data.device)

        if data.dim() == 0 or (data.dim() == 1 and data.numel() == 1):
            # Scalar case
            data = data.view(1)
            times = times.view(1)
            # Interpolate mean and std at times[0]
            indices = torch.searchsorted(time_grid, times, right=False)
            indices = torch.clamp(indices, 0, len(time_grid) - 1)
            t = times[0]
            # Linear interpolation
            if t <= time_grid[0]:
                seq_mean = mean[0]
                seq_std = std[0]
            elif t >= time_grid[-1]:
                # Extrapolate using last two points
                t0, t1 = time_grid[-2], time_grid[-1]
                m0, m1 = mean[-2], mean[-1]
                s0, s1 = std[-2], std[-1]
                seq_mean = m0 + (m1 - m0) * (t - t0) / (t1 - t0)
                seq_std = s0 + (s1 - s0) * (t - t0) / (t1 - t0)
            else:
                # Interpolate within time_grid
                idx = indices[0]
                t0, t1 = time_grid[idx-1], time_grid[idx]
                m0, m1 = mean[idx-1], mean[idx]
                s0, s1 = std[idx-1], std[idx]
                seq_mean = m0 + (m1 - m0) * (t - t0) / (t1 - t0)
                seq_std = s0 + (s1 - s0) * (t - t0) / (t1 - t0)
            result = data * (seq_std + 1e-6) + seq_mean
            return result.squeeze()
        else:
            # Array case
            if data.dim() == 1:
                data = data.unsqueeze(0)
                times = times.unsqueeze(0)
            # Interpolate mean and std for all times
            seq_mean = torch.tensor(np.interp(times.cpu().numpy(), time_grid.cpu().numpy(), mean.cpu().numpy()), device=data.device)
            seq_std = torch.tensor(np.interp(times.cpu().numpy(), time_grid.cpu().numpy(), std.cpu().numpy()), device=data.device)
            result = data * (seq_std + 1e-6) + seq_mean
            if data.shape[0] == 1:
                result = result.squeeze(0)
            return result
    else:
        # NumPy case
        data = np.asarray(data)
        times = np.asarray(times)
        mean = np.asarray(mean)
        std = np.asarray(std)
        time_grid = np.asarray(time_grid)

        if data.size == 1:
            # Scalar case
            data = data.flatten()
            times = times.flatten()
            # Interpolate mean and std at times[0]
            seq_mean = np.interp(times[0], time_grid, mean)
            seq_std = np.interp(times[0], time_grid, std)
            result = data * (seq_std + 1e-6) + seq_mean
            return result[0]
        else:
            # Array case
            # Interpolate mean and std for all times
            seq_mean = np.interp(times, time_grid, mean)
            seq_std = np.interp(times, time_grid, std)
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
        max_length = len(self.current_profiles[0]["current"])
        self.clipped_current_profiles = []

        # Simulate actual voltage trajectory for the first profile
        actual_initial_voltage = 27.4
        self.battery_simulator.set_initial_voltage(actual_initial_voltage, self.current_profiles[0]['current'])
        actual_voltage_trajectory = self.battery_simulator.simulate_battery(self.current_profiles[0]['current'], actual_initial_voltage)
        actual_times = self.current_profiles[0]['time']

        # Sample actual_voltage_trajectory every 50 elements
        sample_interval = 50
        sample_indices = np.arange(0, len(actual_voltage_trajectory), sample_interval)
        num_samples = min(len(sample_indices), len(self.current_profiles))
        initial_voltages = []

        # Standardize sampled voltages
        for i in range(len(self.current_profiles)):
            if i < num_samples:
                sample_idx = sample_indices[i]
                x0 = actual_voltage_trajectory[sample_idx]
                sample_time = actual_times[sample_idx]
            else:
                sample_idx = sample_indices[-1]
                x0 = actual_voltage_trajectory[sample_idx]
                sample_time = actual_times[sample_idx]
            
            x0_standardized = standardize(x0, sample_time, 
                                        self.voltage_mean.cpu().numpy(), 
                                        self.voltage_std.cpu().numpy(), 
                                        self.time_grid.cpu().numpy())
            x0_standardized = torch.tensor(x0_standardized, dtype=torch.float32, device=self.device)
            initial_voltages.append(x0_standardized.item())

        current_inputs = BatteryDataset(self.current_profiles, initial_voltages)
        current_loader = DataLoader(current_inputs, batch_size=1, shuffle=False)
        index_counter = 0

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
                print(len(standardized_current[0].cpu().numpy()))
                print(times[0].cpu().numpy())
                
                plt.plot(standardized_current[0].cpu().numpy())
                plt.show()
                
                # Destandardize the predicted voltage
                voltage = destandardize(output.cpu().numpy(), 
                                        times.cpu().numpy(), 
                                        self.voltage_mean.cpu().numpy(), 
                                        self.voltage_std.cpu().numpy(), 
                                        self.time_grid.cpu().numpy())

            # Store voltage trajectory
            predicted_voltage_trajectories.append({
                "time": times[0].cpu().numpy(), 
                "voltage": voltage[0]
            })

            # Prepare initial voltage for the next prediction
            if idx < len(self.current_profiles) - 1:
                next_initial_time = self.current_profiles[idx + 1]['time'][0]
                time_index = np.searchsorted(times[0].cpu().numpy(), next_initial_time)
                if time_index >= len(output[0]):
                    time_index = len(output[0]) - 1
                index_counter = index_counter + time_index
                initial_voltage_unstandardized = actual_voltage_trajectory[index_counter]
                initial_voltage_standardized = standardize(initial_voltage_unstandardized, 
                                                        actual_times[index_counter], 
                                                        self.voltage_mean.cpu().numpy(), 
                                                        self.voltage_std.cpu().numpy(), 
                                                        self.time_grid.cpu().numpy())
                x0_standardized = torch.tensor(initial_voltage_standardized, dtype=torch.float32, device=self.device).unsqueeze(0)
                initial_voltages[idx + 1] = x0_standardized.item()

            print(f"Prediction {idx}: Time taken = {time.time() - start_time:.2f} seconds")

        with open("saved_predicted_voltage_trajectories.pkl", "wb") as f:
            pickle.dump(predicted_voltage_trajectories, f)
        print("predicted_voltage_trajectories saved to 'predicted_voltage_trajectories.pkl'.")

        with open("saved_clipped_current_profiles.pkl", "wb") as f:
            pickle.dump(self.clipped_current_profiles, f)
        print("clipped_current_profiles saved to 'clipped_current_profiles.pkl'.")        

        return predicted_voltage_trajectories

    def predict_voltage_new(self):
        predicted_voltage_trajectories = []
        max_length = len(self.current_profiles[0]["current"])
        self.clipped_current_profiles = []

        # Simulate voltage trajectory for the first profile
        actual_initial_voltage = 27.4
        first_profile = self.current_profiles[0]
        actual_start_time = first_profile['time'][0]
        
        # Create combined current profile for simulation
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

        # Interpolate combined profile to 0.1-second step for simulation
        if len(combined_time) > 1:
            new_combined_time = np.arange(combined_time[0], combined_time[-1] + 0.1, 0.1)
            new_combined_current = np.interp(new_combined_time, combined_time, combined_current)
        else:
            new_combined_time = combined_time
            new_combined_current = combined_current

        # Simulate battery
        # self.battery_simulator.set_initial_voltage(actual_initial_voltage, new_combined_current)
        simulated_voltage = self.battery_simulator.simulate_battery(new_combined_current, actual_initial_voltage)
        simulated_time = new_combined_time
        # print(len(new_combined_current))
        # plt.plot(simulated_time, new_combined_current)
        # plt.show()
        # plt.plot(simulated_time, simulated_voltage)


        # Sample initial voltages every 50 points from actual_start_time
        initial_voltages = []
        clip_idx = np.searchsorted(simulated_time, actual_start_time)
        simulated_voltage = simulated_voltage[clip_idx:]
        simulated_time = simulated_time[clip_idx:]
        
        sample_interval = 50
        sample_indices = np.arange(0, len(simulated_voltage), sample_interval)
        num_samples = min(len(sample_indices), len(self.current_profiles))
        
        for i in range(len(self.current_profiles)):
            if i < num_samples:
                sample_idx = sample_indices[i]
            else:
                sample_idx = sample_indices[-1]
            voltage = simulated_voltage[sample_idx]
            sample_time = simulated_time[sample_idx]
            
            # Standardize voltage
            standardized_voltage = standardize(
                voltage, sample_time, 
                self.voltage_mean.cpu().numpy(), 
                self.voltage_std.cpu().numpy(), 
                self.time_grid.cpu().numpy()
            )
            initial_voltages.append(torch.tensor(standardized_voltage, dtype=torch.float32, device=self.device).item())

        current_inputs = BatteryDataset(self.current_profiles, initial_voltages)
        current_loader = DataLoader(current_inputs, batch_size=1, shuffle=False)
        index_counter = 0

        for idx, (current_profile, time_horizon, init_vol) in enumerate(current_loader):
            start_time = time.time()

            # Move tensors to device
            current_profile = current_profile.to(self.device)
            time_horizon = time_horizon.to(self.device)
            init_vol = init_vol.to(self.device)

            # Create combined current profile
            actual_start_time = time_horizon[0, 0].item()
            if actual_start_time > 0:
                dummy_time = np.arange(0, actual_start_time + 5, 5)
                if dummy_time[-1] < actual_start_time:
                    dummy_time = np.append(dummy_time, actual_start_time)
                dummy_current = np.full_like(dummy_time, 64.6)
                combined_time = np.concatenate([dummy_time, time_horizon[0].cpu().numpy()])
                combined_current = np.concatenate([dummy_current, current_profile[0].cpu().numpy()])
            else:
                combined_time = time_horizon[0, :max_length].cpu().numpy()
                combined_current = current_profile[0, :max_length].cpu().numpy()

            # Interpolate to 0.1-second time step
            if len(combined_time) > 1:
                new_times_np = np.arange(combined_time[0], combined_time[-1] + 0.1, 0.1)
                new_current_np = np.interp(new_times_np, combined_time, combined_current)
            else:
                new_times_np = combined_time
                new_current_np = combined_current

            # Convert to tensors
            times = torch.tensor(new_times_np, dtype=torch.float32, device=self.device).unsqueeze(0)
            current_values = torch.tensor(new_current_np, dtype=torch.float32, device=self.device).unsqueeze(0)

            # Save complete interpolated profile
            self.clipped_current_profiles.append({
                "current": new_current_np[clip_idx:],
                "time": new_times_np[clip_idx:]
            })

            # Standardize interpolated current
            standardized_current = torch.tensor(
                standardize(current_values.cpu().numpy(), times.cpu().numpy(), 
                            self.current_mean.cpu().numpy(), self.current_std.cpu().numpy(), 
                            self.time_grid.cpu().numpy()),
                dtype=torch.float32, device=self.device
            )

            # Model prediction
            with torch.no_grad():
                output = self.model(standardized_current, times, init_vol)
                output = output.squeeze(-1)

            # Destandardize voltage
            voltage = destandardize(output, times, self.voltage_mean, self.voltage_std, self.time_grid)
            voltage = voltage.cpu().numpy() if isinstance(voltage, torch.Tensor) else voltage
            
            # plt.plot(times[0].cpu().numpy(), voltage)
            # plt.show()

            # Clip and interpolate voltage trajectory
            clip_idx = np.searchsorted(times[0].cpu().numpy(), actual_start_time)
            clipped_time = times[0, clip_idx:].cpu().numpy()
            clipped_voltage = voltage[clip_idx:]
            
            if len(clipped_time) > 1:
                new_clipped_time = np.arange(clipped_time[0], clipped_time[-1] + 0.1, 0.1)
                new_clipped_voltage = np.interp(new_clipped_time, clipped_time, clipped_voltage)
            else:
                new_clipped_time = clipped_time
                new_clipped_voltage = clipped_voltage

            # Store interpolated voltage trajectory
            predicted_voltage_trajectories.append({
                "time": new_clipped_time,
                "voltage": new_clipped_voltage
            })

            # Prepare initial voltage for next prediction
            if idx < len(self.current_profiles) - 1:
                next_initial_time = self.current_profiles[idx + 1]['time'][0]
                time_index = np.searchsorted(times[0].cpu().numpy(), next_initial_time)
                if time_index >= len(output[0]):
                    time_index = len(output[0]) - 1
                index_counter += time_index
                if index_counter >= len(simulated_voltage):
                    index_counter = len(simulated_voltage) - 1
                initial_voltage_unstandardized = simulated_voltage[index_counter]
                initial_voltage_standardized = standardize(
                    initial_voltage_unstandardized, simulated_time[index_counter], 
                    self.voltage_mean.cpu().numpy(), self.voltage_std.cpu().numpy(), 
                    self.time_grid.cpu().numpy()
                )
                x0_standardized = torch.tensor(initial_voltage_standardized, dtype=torch.float32, device=self.device).unsqueeze(0)
                initial_voltages[idx + 1] = x0_standardized.item()

            print(f"Prediction {idx}: Time taken = {time.time() - start_time:.2f} seconds")

        # Save profiles
        with open("new_voltages/saved_clipped_current_profiles.pkl", "wb") as f:
            pickle.dump(self.clipped_current_profiles, f)
        print("clipped_current_profiles saved to 'new_voltages/saved_clipped_current_profiles.pkl'.")

        with open("new_voltages/saved_predicted_voltage_trajectories.pkl", "wb") as f:
            pickle.dump(predicted_voltage_trajectories, f)
        print("predicted_voltage_trajectories saved to 'new_voltages/saved_predicted_voltage_trajectories.pkl'.")

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
            print(predicted_initial_voltage)
            # print(time)
            # plt.plot(time,current)
            # plt.show()
            
       

            self.battery_simulator.set_initial_voltage(predicted_initial_voltage, current)
            voltage_trajectory = self.battery_simulator.simulate_battery(current, predicted_initial_voltage)


            if voltage_trajectory is not None:
                actual_voltage_trajectories.append({
                    "time": time.tolist(), 
                    "voltage": voltage_trajectory.tolist()
                })
                
        with open("new_voltages/saved_actual_voltage_trajectories.pkl", "wb") as f:
            pickle.dump(actual_voltage_trajectories, f)
        print("actual_voltage_trajectories saved to 'new_voltages/actual_voltage_trajectories.pkl'.")

        return actual_voltage_trajectories


    def plot_comparison_voltage_trajectories(self, predicted_voltage_trajectories, actual_voltage_trajectories):
        fig, ax = plt.subplots(figsize=(5, 4), dpi=300)
        num_trajectories = min(50, len(predicted_voltage_trajectories))
        colors = plt.cm.viridis(np.linspace(0, 1, num_trajectories))
        time_shift = 0
        batt_calibration = 2

        for idx, (predicted, actual) in enumerate(zip(predicted_voltage_trajectories[:50], actual_voltage_trajectories[:50])):
            pred_time, pred_voltage = np.array(predicted["time"]), np.array(predicted["voltage"])  + 0.318
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
            ax.plot(first_pred["time"] + time_shift, np.array(first_pred["voltage"]) - batt_calibration + 0.318, 
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

if __name__ == "__main__":
    BASE_DIR = Path(__file__).resolve().parents[1]
    TRAIN_PATH = BASE_DIR /"data" /"training_data_mixed.pkl"
    TEST_PATH = BASE_DIR /"data"/ "test_data_mixed.pkl"
    MODEL_PATH = BASE_DIR /"saved_models"/ 'trained_pinn_model_actual.pth'
    CURRENT_PROFILE_PATH = BASE_DIR /"data" /"SimulationResults" /"new_sim_data"/ "saved_current_profiles_EM2.pkl"
    predictor = VoltagePredictor(MODEL_PATH, CURRENT_PROFILE_PATH, TRAIN_PATH, TEST_PATH)
    predicted_voltage_trajectories = predictor.predict_voltage_new()
    actual_voltage_trajectories = predictor.simulate_actual_battery(predicted_voltage_trajectories)
    predicted_voltage_trajectories = predictor.load_saved_current_profiles(BASE_DIR/"new_voltages" /"saved_predicted_voltage_trajectories.pkl")
    actual_voltage_trajectories = predictor.load_saved_current_profiles(BASE_DIR/ "new_voltages" /"saved_actual_voltage_trajectories.pkl")
    predictor.clipped_current_profiles = predictor.load_saved_current_profiles(BASE_DIR/ "new_voltages"/ "saved_clipped_current_profiles.pkl")
    predictor.plot_comparison_voltage_trajectories(predicted_voltage_trajectories, actual_voltage_trajectories)