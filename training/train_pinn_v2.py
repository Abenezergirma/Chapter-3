import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm

class BatteryDataset(Dataset):
    def __init__(self, inputs, outputs, times, initial_voltages):
        self.inputs = inputs
        self.outputs = outputs
        self.times = times
        self.initial_voltages = initial_voltages

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.outputs[idx], self.times[idx], self.initial_voltages[idx]

def custom_collate_fn(batch):
    inputs, outputs, times, initial_voltages = zip(*batch)
    max_len = max(len(seq) for seq in inputs)
    inputs_padded = torch.stack([
        torch.tensor(np.pad(seq, (0, max_len - len(seq)), mode='constant', constant_values=0), dtype=torch.float32)
        for seq in inputs
    ])
    outputs_padded = torch.stack([
        torch.tensor(np.pad(seq, (0, max_len - len(seq)), mode='constant', constant_values=0), dtype=torch.float32)
        for seq in outputs
    ])
    times_padded = torch.stack([
        torch.tensor(np.pad(seq, (0, max_len - len(seq)), mode='constant', constant_values=0), dtype=torch.float32)
        for seq in times
    ])
    initial_voltages = torch.tensor(initial_voltages, dtype=torch.float32)
    return inputs_padded, outputs_padded, times_padded, initial_voltages

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

def physics_loss(predicted, true, current, times, initial_voltages, params):
    if not params:
        return 0
    R0 = params.get('R0', 0.01311)
    R1 = params.get('R1', 0.09957)
    C1 = params.get('C1', 3824.34)
    output_mean = params['output_mean']
    output_std = params['output_std']
    input_mean = params['input_mean']
    input_std = params['input_std']
    time_grid = params['time_grid']

    V0 = destandardize(initial_voltages, times[:, 0], output_mean, output_std, time_grid)
    V_pred = destandardize(predicted, times, output_mean, output_std, time_grid)
    I_actual = destandardize(current, times, input_mean, input_std, time_grid)

    if not hasattr(physics_loss, "current_plotted"):
        valid_len = (times[0] != 0).sum().item()
        plt.figure(figsize=(10, 5))
        plt.plot(times[0, :valid_len].detach().numpy(), current[0, :valid_len].cpu().numpy(), label='Standardized Current', color='green')
        plt.plot(times[0, :valid_len].detach().numpy(), I_actual[0, :valid_len].cpu().numpy(), label='Destandardized Current', color='red')
        plt.xlabel('Time (s)')
        plt.ylabel('Current (A)')
        plt.legend()
        plt.savefig('current_physics_loss.png', dpi=100, bbox_inches='tight')
        plt.close()
        physics_loss.current_plotted = True

    if not hasattr(physics_loss, "debug_printed"):
        valid_len = (times[0] != 0).sum().item()
        print(f"\nDebugging destandardization in physics_loss (first sequence in batch):")
        print(f"Valid length: {valid_len}, Total length: {times.shape[1]}")
        for i in range(max(0, valid_len - 2), min(times.shape[1], valid_len + 2)):
            t = times[0, i]
            curr = float(current[0, i].item())
            i_act = float(I_actual[0, i].item())
            idx = torch.argmin(torch.abs(time_grid - t)).item()
            mean_val = float(input_mean[idx].item())
            std_val = float(input_std[idx].item())
            print(f"Timestep {i} (t={t.item():.1f}, time_grid[{idx}]={time_grid[idx].item():.1f}):")
            print(f"  Standardized Current: {curr:.4f}")
            print(f"  Mean: {mean_val:.4f}, Std: {std_val:.4f}")
            print(f"  Destandardized Current: {i_act:.4f}")
        physics_loss.debug_printed = True

    V_thevenin = simulate_thevenin(I_actual, times, R0, R1, C1, V0)
    V_thevenin = torch.tensor(V_thevenin, device=predicted.device, dtype=torch.float32)

    if not hasattr(physics_loss, "plotted"):
        valid_len = (times[0] != 0).sum().item()
        plt.figure(figsize=(10, 5))
        plt.plot(times[0, :valid_len].detach().numpy(), V_pred[0, :valid_len].detach().numpy(), label='Data Loss (Destandardized)', color='blue')
        plt.plot(times[0, :valid_len].detach().numpy(), V_thevenin[0, :valid_len].detach().numpy(), label='Physics Loss (Thevenin)', color='orange')
        plt.xlabel('Time (s)')
        plt.ylabel('Voltage (V)')
        plt.legend()
        plt.savefig('voltage_physics_loss.png', dpi=100, bbox_inches='tight')
        plt.close()
        physics_loss.plotted = True

    return torch.mean((V_pred - V_thevenin) ** 2)

def simulate_thevenin(I, t, R0, R1, C1, V0):
    if V0.dim() == 1:
        V0 = V0.unsqueeze(-1)
    V0 = V0.expand(-1, I.shape[1])

    V_R1 = torch.zeros_like(I[:, :1])
    dt = torch.diff(t, dim=1, prepend=t[:, :1])

    V = [V0[:, :1]]
    for i in range(1, I.shape[1]):
        dV_R1 = (-(1 / (R1 * C1)) * V_R1 + I[:, i-1:i] / C1) * dt[:, i:i+1]
        V_R1 += dV_R1
        V_t = V0[:, i:i+1] - I[:, i:i+1] * R0 - V_R1
        V.append(V_t)

    return torch.cat(V, dim=1)

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
    full_train_outputs = [seq for seq in train_outputs if len(seq) == max_len]
    full_train_times = [t for t in train_times if len(t) == max_len]

    train_inputs, input_mean, input_std, time_grid = timestep_standardize(
        train_inputs, train_times, full_data=full_train_inputs, full_times=full_train_times
    )
    train_outputs, output_mean, output_std, _ = timestep_standardize(
        train_outputs, train_times, full_data=full_train_outputs, full_times=full_train_times
    )
    train_initial_voltages = [v[0] for v in train_outputs]

    test_inputs = [standardize(seq, t, input_mean, input_std, time_grid) for seq, t in zip(test_inputs, test_times)]
    test_outputs = [standardize(seq, t, output_mean, output_std, time_grid) for seq, t in zip(test_outputs, test_times)]
    test_initial_voltages = [
        standardize(v[0], t[0], output_mean, output_std, time_grid) for v, t in zip(test_outputs, test_times)
    ]

    try:
        fig, axs = plt.subplots(3, 2, figsize=(16, 9))
        full_idx = next(i for i, seq in enumerate(train_inputs) if len(seq) == max_len)
        raw_input = np.array(training_data[full_idx]['input'])
        std_input = train_inputs[full_idx]
        destd_input = destandardize(torch.tensor(std_input), torch.tensor(train_times[full_idx]), 
                                   torch.tensor(input_mean), torch.tensor(input_std), torch.tensor(time_grid)).numpy()
        
        print(f"Full trajectory time range: {train_times[full_idx][0]} to {train_times[full_idx][-1]}")
        print(f"Full trajectory length: {len(train_times[full_idx])}")

        downsample_factor = 50
        full_times_downsampled = train_times[full_idx][::downsample_factor]
        raw_input_downsampled = raw_input[::downsample_factor]
        std_input_downsampled = std_input[::downsample_factor]
        destd_input_downsampled = destd_input[::downsample_factor]

        print(f"Downsampled full trajectory length: {len(full_times_downsampled)}")

        axs[0, 0].plot(full_times_downsampled, raw_input_downsampled, label='Raw Input (Full)')
        axs[0, 0].set_title('Raw Current (Full Trajectory)')
        axs[0, 0].legend()
        axs[0, 0].set_aspect('auto')
        
        axs[1, 0].plot(full_times_downsampled, std_input_downsampled, label='Standardized Input (Full)')
        axs[1, 0].set_title('Standardized Current (Full Trajectory)')
        axs[1, 0].legend()
        axs[1, 0].set_aspect('auto')
        
        axs[2, 0].plot(full_times_downsampled, destd_input_downsampled, label='Destandardized Input (Full)')
        axs[2, 0].plot(full_times_downsampled, raw_input_downsampled, label='Raw Input (Full)', linestyle='--')
        axs[2, 0].set_title('Destandardized vs Raw Current (Full Trajectory)')
        axs[2, 0].legend()
        axs[2, 0].set_aspect('auto')

        mid_idx = next(i for i, seq in enumerate(train_inputs) if len(seq) < max_len)
        raw_input_mid = np.array(training_data[mid_idx]['input'])
        std_input_mid = train_inputs[mid_idx]
        destd_input_mid = destandardize(torch.tensor(std_input_mid), torch.tensor(train_times[mid_idx]), 
                                       torch.tensor(input_mean), torch.tensor(input_std), torch.tensor(time_grid)).numpy()
        
        print(f"Mid-flight trajectory time range: {train_times[mid_idx][0]} to {train_times[mid_idx][-1]}")
        print(f"Mid-flight trajectory length: {len(train_times[mid_idx])}")

        mid_times = train_times[mid_idx]
        raw_input_mid_plot = raw_input_mid
        std_input_mid_plot = std_input_mid
        destd_input_mid_plot = destd_input_mid
        if len(mid_times) > 1000:
            downsample_factor_mid = len(mid_times) // 1000 + 1
            mid_times = mid_times[::downsample_factor_mid]
            raw_input_mid_plot = raw_input_mid[::downsample_factor_mid]
            std_input_mid_plot = std_input_mid[::downsample_factor_mid]
            destd_input_mid_plot = destd_input_mid[::downsample_factor_mid]

        print(f"Downsampled mid-flight trajectory length: {len(mid_times)}")

        axs[0, 1].plot(mid_times, raw_input_mid_plot, label='Raw Input (Mid)')
        axs[0, 1].set_title('Raw Current (Mid-Flight Trajectory)')
        axs[0, 1].legend()
        axs[0, 1].set_aspect('auto')
        
        axs[1, 1].plot(mid_times, std_input_mid_plot, label='Standardized Input (Mid)')
        axs[1, 1].set_title('Standardized Current (Mid-Flight Trajectory)')
        axs[1, 1].legend()
        axs[1, 1].set_aspect('auto')
        
        axs[2, 1].plot(mid_times, destd_input_mid_plot, label='Destandardized Input (Mid)')
        axs[2, 1].plot(mid_times, raw_input_mid_plot, label='Raw Input (Mid)', linestyle='--')
        axs[2, 1].set_title('Destandardized vs Raw Current (Mid-Flight Trajectory)')
        axs[2, 1].legend()
        axs[2, 1].set_aspect('auto')

        plt.tight_layout()
        plt.savefig('trajectory_plots.png', dpi=100, bbox_inches='tight')
        print("Plot saved as 'trajectory_plots.png'")
        plt.close(fig)

    except Exception as e:
        print(f"Error while generating plots: {e}")
        print("Skipping plot generation to continue execution.")

    num_steps_to_check = 5
    print(f"\nDebugging last {num_steps_to_check} timesteps of mid-flight trajectory (index {mid_idx}):")
    print(f"Time range: {train_times[mid_idx][0]} to {train_times[mid_idx][-1]}")
    print(f"Shape of destd_input_mid: {destd_input_mid.shape}")
    for i in range(-num_steps_to_check, 0):
        t = train_times[mid_idx][i]
        idx = np.searchsorted(time_grid, t, side='left')
        idx = np.clip(idx, 0, len(time_grid) - 1)
        raw_val = float(raw_input_mid[i])
        std_val = float(std_input_mid[i])
        destd_val = float(destd_input_mid[i])
        mean_val = float(input_mean[idx])
        std_val_used = float(input_std[idx])
        print(f"Timestep {i} (t={t:.1f}, time_grid[{idx}]={time_grid[idx]:.1f}):")
        print(f"  Raw Current: {raw_val:.4f}")
        print(f"  Mean: {mean_val:.4f}, Std: {std_val_used:.4f}")
        print(f"  Standardized Current: {std_val:.4f}")
        print(f"  Destandardized Current: {destd_val:.4f}")

    mid_time_range = (train_times[mid_idx][0], train_times[mid_idx][-1])
    mid_time_mask = (time_grid >= mid_time_range[0]) & (time_grid <= mid_time_range[1])

    try:
        fig, axs = plt.subplots(3, 2, figsize=(16, 9))
        downsample_factor_stats = 10
        time_grid_downsampled = time_grid[::downsample_factor_stats]
        input_mean_downsampled = input_mean[::downsample_factor_stats]
        input_std_downsampled = input_std[::downsample_factor_stats]
        output_mean_downsampled = output_mean[::downsample_factor_stats]
        output_std_downsampled = output_std[::downsample_factor_stats]

        axs[0, 0].plot(time_grid_downsampled, input_mean_downsampled, label='Mean of Inputs')
        axs[0, 0].plot(time_grid_downsampled, input_std_downsampled, label='Std Dev of Inputs')
        axs[0, 0].set_title('Input Statistics (Full Trajectories)')
        axs[0, 0].legend()
        axs[0, 0].set_aspect('auto')

        axs[0, 1].plot(time_grid_downsampled, output_mean_downsampled, label='Mean of Outputs')
        axs[0, 1].plot(time_grid_downsampled, output_std_downsampled, label='Std Dev of Outputs')
        axs[0, 1].set_title('Output Statistics (Full Trajectories)')
        axs[0, 1].legend()
        axs[0, 1].set_aspect('auto')

        mid_time_mask_downsampled = (time_grid_downsampled >= mid_time_range[0]) & (time_grid_downsampled <= mid_time_range[1])
        axs[1, 0].plot(time_grid_downsampled[mid_time_mask_downsampled], input_mean_downsampled[mid_time_mask_downsampled], label='Mean of Inputs (Zoomed)')
        axs[1, 0].plot(time_grid_downsampled[mid_time_mask_downsampled], input_std_downsampled[mid_time_mask_downsampled], label='Std Dev of Inputs (Zoomed)')
        axs[1, 0].set_title(f'Input Statistics (t={mid_time_range[0]} to {mid_time_range[1]})')
        axs[1, 0].legend()
        axs[1, 0].set_aspect('auto')

        axs[1, 1].plot(time_grid_downsampled[mid_time_mask_downsampled], output_mean_downsampled[mid_time_mask_downsampled], label='Mean of Outputs (Zoomed)')
        axs[1, 1].plot(time_grid_downsampled[mid_time_mask_downsampled], output_std_downsampled[mid_time_mask_downsampled], label='Std Dev of Outputs (Zoomed)')
        axs[1, 1].set_title(f'Output Statistics (t={mid_time_range[0]} to {mid_time_range[1]})')
        axs[1, 1].legend()
        axs[1, 1].set_aspect('auto')

        train_times_0 = train_times[0]
        train_inputs_0 = train_inputs[0]
        train_outputs_0 = train_outputs[0]
        if len(train_times_0) > 1000:
            downsample_factor_sample = len(train_times_0) // 1000 + 1
            train_times_0 = train_times_0[::downsample_factor_sample]
            train_inputs_0 = train_inputs_0[::downsample_factor_sample]
            train_outputs_0 = train_outputs_0[::downsample_factor_sample]

        axs[2, 0].plot(train_times_0, train_inputs_0, label='Sample Standardized Input')
        axs[2, 0].set_title('Sample Standardized Input Visualization')
        axs[2, 0].legend()
        axs[2, 0].set_aspect('auto')

        axs[2, 1].plot(train_times_0, train_outputs_0, label='Sample Standardized Output')
        axs[2, 1].set_title('Sample Standardized Output Visualization')
        axs[2, 1].legend()
        axs[2, 1].set_aspect('auto')

        plt.tight_layout()
        plt.savefig('statistics_plots.png', dpi=100, bbox_inches='tight')
        print("Plot saved as 'statistics_plots.png'")
        plt.close(fig)

    except Exception as e:
        print(f"Error while generating statistics plots: {e}")
        print("Skipping statistics plot generation to continue execution.")

    train_dataset = BatteryDataset(train_inputs, train_outputs, train_times, train_initial_voltages)
    test_dataset = BatteryDataset(test_inputs, test_outputs, test_times, test_initial_voltages)
    return train_dataset, test_dataset, (input_mean, input_std), (output_mean, output_std), time_grid

def train_pinn_model(model, train_loader, test_loader, epochs, lr, use_physics_loss=False, physical_params=None):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    device = next(model.parameters()).device

    if physical_params:
        physical_params['input_mean'] = torch.tensor(physical_params['input_mean'], dtype=torch.float32, device=device)
        physical_params['input_std'] = torch.tensor(physical_params['input_std'], dtype=torch.float32, device=device)
        physical_params['output_mean'] = torch.tensor(physical_params['output_mean'], dtype=torch.float32, device=device)
        physical_params['output_std'] = torch.tensor(physical_params['output_std'], dtype=torch.float32, device=device)
        physical_params['time_grid'] = torch.tensor(physical_params['time_grid'], dtype=torch.float32, device=device)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for inputs, targets, times, initial_voltages in tqdm(train_loader, desc=f"Training Epoch {epoch+1}"):
            inputs = inputs.to(device)
            targets = targets.to(device)
            times = times.to(device)
            initial_voltages = initial_voltages.to(device)

            optimizer.zero_grad()
            outputs = model(inputs, times, initial_voltages)
            outputs = outputs.squeeze(-1)

            initial_loss = torch.mean((outputs[:, 0] - initial_voltages) ** 2)
            loss = criterion(outputs, targets) + 0.5 * initial_loss

            if use_physics_loss and physical_params:
                phys_loss = physics_loss(outputs, targets, inputs, times, initial_voltages, physical_params)
                loss += phys_loss

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f'Epoch {epoch+1}, Loss: {total_loss / len(train_loader)}')

        model.eval()
        with torch.no_grad():
            test_loss = 0
            for inputs, targets, times, initial_voltages in tqdm(test_loader, desc=f"Evaluating Epoch {epoch+1}"):
                inputs = inputs.to(device)
                targets = targets.to(device)
                times = times.to(device)
                initial_voltages = initial_voltages.to(device)
                outputs = model(inputs, times, initial_voltages)
                outputs = outputs.squeeze(-1)
                test_loss += criterion(outputs, targets).item()
            print(f'Test Loss: {test_loss / len(test_loader)}')

    torch.save(model.state_dict(), 'trained_pinn_model_actual.pth')

def evaluate_model(model_path, test_data_path, model_class, hidden_size, output_size, output_mean, output_std, time_grid):
    model = model_class(hidden_size=hidden_size, output_size=output_size)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    device = next(model.parameters()).device
    output_mean = torch.tensor(output_mean, dtype=torch.float32, device=device)
    output_std = torch.tensor(output_std, dtype=torch.float32, device=device)
    time_grid = torch.tensor(time_grid, dtype=torch.float32, device=device)

    with open(test_data_path, "rb") as f:
        test_data = pickle.load(f)

    test_inputs = [np.array(d['input']) for d in test_data]
    test_outputs = [np.array(d['output']) for d in test_data]
    test_times = [np.array(d['time']) for d in test_data]
    test_initial_voltages = [v[0] for v in test_outputs]

    test_inputs = [standardize(seq, t, input_mean, input_std, time_grid.cpu().numpy()) for seq, t in zip(test_inputs, test_times)]
    test_outputs = [standardize(seq, t, output_mean.cpu().numpy(), output_std.cpu().numpy(), time_grid.cpu().numpy()) for seq, t in zip(test_outputs, test_times)]
    test_initial_voltages = [
        standardize(v[0], t[0], output_mean.cpu().numpy(), output_std.cpu().numpy(), time_grid.cpu().numpy()) for v, t in zip(test_outputs, test_times)
    ]

    test_dataset = BatteryDataset(test_inputs, test_outputs, test_times, test_initial_voltages)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=custom_collate_fn)

    all_predictions = []
    all_targets = []
    all_initial_voltages = []
    all_times = []
    for inputs, targets, times, initial_voltages in tqdm(test_loader):
        with torch.no_grad():
            inputs = inputs.to(device)
            targets = targets.to(device)
            times = times.to(device)
            initial_voltages = initial_voltages.to(device)
            outputs = model(inputs, times, initial_voltages)
            outputs = outputs.squeeze(-1)
            outputs_destd = destandardize(outputs, times, output_mean, output_std, time_grid)
            targets_destd = destandardize(targets, times, output_mean, output_std, time_grid)
            all_predictions.append(outputs_destd.cpu().numpy())
            all_targets.append(targets_destd.cpu().numpy())
            all_initial_voltages.append(
                destandardize(initial_voltages, times[:, 0], output_mean, output_std, time_grid).cpu().numpy()
            )
            all_times.append(times.cpu().numpy())

    try:
        plt.figure(figsize=(10, 5))
        for i in range(min(3, len(test_times))):
            valid_len = len(test_times[i]) - (test_times[i] == 0).sum()
            times_i = all_times[i][0, :valid_len]
            predictions_i = all_predictions[i][0, :valid_len]
            targets_i = all_targets[i][0, :valid_len]
            if len(times_i) > 1000:
                downsample_factor = len(times_i) // 1000 + 1
                times_i = times_i[::downsample_factor]
                predictions_i = predictions_i[::downsample_factor]
                targets_i = targets_i[::downsample_factor]
            plt.plot(times_i, predictions_i, label=f'Predicted (V0={all_initial_voltages[i][0]:.2f}V)')
            plt.plot(times_i, targets_i, linestyle='--', label=f'True (V0={all_initial_voltages[i][0]:.2f}V)')
        plt.title('Voltage Trajectories with Variable Initial Voltages and Time Horizons')
        plt.xlabel('Time (s)')
        plt.ylabel('Voltage (V)')
        plt.legend()
        plt.savefig('evaluation_plots.png', dpi=100, bbox_inches='tight')
        print("Evaluation plot saved as 'evaluation_plots.png'")
        plt.close()

    except Exception as e:
        print(f"Error while generating evaluation plots: {e}")
        print("Skipping evaluation plot generation to continue execution.")

if __name__ == "__main__":
    BASE_DIR = Path(__file__).resolve().parent.parent
    TRAIN_PATH = BASE_DIR /  "data" / "training_data_mixed.pkl"
    TEST_PATH = BASE_DIR /  "data" /"test_data_mixed.pkl"
    MODEL_PATH = BASE_DIR / 'trained_pinn_model_actual.pth'

    train_dataset, test_dataset, (input_mean, input_std), (output_mean, output_std), time_grid = load_data(TRAIN_PATH, TEST_PATH)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=custom_collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=custom_collate_fn)

    model = LSTMPINNBatteryModel(hidden_size=128, output_size=1)
    physical_params = {
        'R0': 0.01311,
        'R1': 0.09957,
        'C1': 3824.34,
        'input_mean': input_mean,
        'input_std': input_std,
        'output_mean': output_mean,
        'output_std': output_std,
        'time_grid': time_grid
    }
    # train_pinn_model(model, train_loader, test_loader, epochs=100, lr=0.001, use_physics_loss=False, physical_params=physical_params)
    evaluate_model(MODEL_PATH, TEST_PATH, LSTMPINNBatteryModel, 128, 1, output_mean, output_std, time_grid)