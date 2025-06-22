import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchdiffeq import odeint
from tqdm import tqdm
from pathlib import Path
import matplotlib.pyplot as plt

def standardize(data, mean, std):
    return (data - mean) / std

def destandardize(data, mean, std):
    return data * std + mean

def combined_mse_rmse_loss(output, target):
    mse = torch.mean((output - target) ** 2)
    rmse = torch.sqrt(mse)
    return mse + rmse


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
        # print(f"[DEBUG] current_profile shape: {self.current_profile.shape} | Expected: [B, T, 1]")

        # Use the full time tensor and figure out the index of the current scalar t
        t_all = self.time_tensor  # shape: [T]
        if t_all.dim() != 1:
            raise ValueError(f"[ERROR] Expected time_tensor to be 1D, got {t_all.shape}")

        # Find the matching index in the time array (closest to scalar t)
        t_scalar = t.item()
        time_idx = torch.argmin(torch.abs(t_all - t_scalar)).item()
        # print(f"[DEBUG] Using stored time_tensor to find closest index to t={t_scalar:.5f} → index={time_idx}")

        current_at_t = self.current_profile[:, time_idx, :]  # Shape: [B, 1]
        # print(f"[DEBUG] current_at_t shape: {current_at_t.shape} | Expected: [B, 1]")

        time_tensor = t.expand(batch_size, 1)  # [B, 1]
        # print(f"[DEBUG] expanded t shape: {time_tensor.shape} | Expected: [B, 1]")

        # print(f"[DEBUG] x shape: {x.shape} | Expected: [B, 1]")

        input_tensor = torch.cat((x, current_at_t, time_tensor), dim=-1)  # [B, 3]
        # print(f"[DEBUG] input_tensor shape: {input_tensor.shape} | Expected: [B, 3]")

        output = self.net(input_tensor)
        # print(f"[DEBUG] output shape from net: {output.shape} | Expected: [B, 1]")

        return output




class NeuralODE(nn.Module):
    def __init__(self, state_dim, hidden_dim):
        super(NeuralODE, self).__init__()
        self.ode_func = ODEFunc(state_dim, hidden_dim)

    def forward(self, x0, current_profile, t):
        self.ode_func.set_current_profile(current_profile)
        self.ode_func.set_time_tensor(t)
        out = odeint(self.ode_func, x0, t, method="rk4", rtol=1e-3, atol=1e-4)
        return out.permute(1, 0, 2)


# Load and preprocess training/test data with time-step-specific standardization
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

    # Avoid division by zero
    input_std[input_std == 0] = 1
    output_std[output_std == 0] = 1

    def preprocess(data, mean, std):
        standardized = [(d - mean) / std for d in data]
        return [torch.tensor(d, dtype=torch.float32).unsqueeze(-1) for d in standardized]

    train_inputs = preprocess([d["input"] for d in training_data], input_mean, input_std)
    train_outputs = preprocess([d["output"] for d in training_data], output_mean, output_std)
    test_inputs = preprocess([d["input"] for d in test_data], input_mean, input_std)
    test_outputs = preprocess([d["output"] for d in test_data], output_mean, output_std)

    # Visualize standardization
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    axs[0, 0].plot(input_mean, label='Mean of Inputs')
    axs[0, 0].plot(input_std, label='Std Dev of Inputs')
    axs[0, 0].set_title('Input Statistics')
    axs[0, 0].legend()

    axs[0, 1].plot(output_mean, label='Mean of Outputs')
    axs[0, 1].plot(output_std, label='Std Dev of Outputs')
    axs[0, 1].set_title('Output Statistics')
    axs[0, 1].legend()

    axs[1, 0].plot(train_inputs[0].squeeze(), label='Sample Standardized Input')
    axs[1, 0].set_title('Sample Standardized Input Visualization')
    axs[1, 0].legend()

    axs[1, 1].plot(train_outputs[0].squeeze(), label='Sample Standardized Output')
    axs[1, 1].set_title('Sample Standardized Output Visualization')
    axs[1, 1].legend()

    plt.tight_layout()
    plt.show()

    return train_inputs, train_outputs, test_inputs, test_outputs, (input_mean, input_std), (output_mean, output_std)


def create_data_loader(inputs, outputs, batch_size=4):
    dataset = torch.utils.data.TensorDataset(torch.stack(inputs), torch.stack(outputs))
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

def train_neural_ode(model, train_loader, num_epochs=300, learning_rate=0.001):
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for batch_inputs, batch_outputs in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}'):
            optimizer.zero_grad()
            x0 = batch_outputs[:, 0, :]  # Initialize x0
            # print(x0)
            t = torch.linspace(0, 1, batch_inputs.size(1))   # Set the time tensor
            pred_outputs = model(x0, batch_inputs, t)
            loss = combined_mse_rmse_loss(pred_outputs, batch_outputs)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        scheduler.step(avg_loss)
        print(f"Epoch {epoch+1}, Loss: {avg_loss:.6f}")

        if (epoch + 1) % 50 == 0:
            torch.save(model.state_dict(), f"neural_ode_epoch{epoch+1}.pth")

def evaluate_model(model, test_inputs, test_outputs, input_mean, input_std, output_mean, output_std,
                   num_trajectories=5, time_step=0.1, preprocess_inputs=False):
    """
    Evaluates the trained model on a few test trajectories and plots the predicted vs actual voltages.

    Parameters:
        - model: Trained NeuralODE model
        - test_inputs: List of preprocessed input tensors (standardized current)
        - test_outputs: List of preprocessed output tensors (standardized voltage)
        - input_mean, input_std: Needed if inputs need to be standardized on-the-fly
        - output_mean, output_std: For de-standardizing the predicted/true outputs
        - num_trajectories: Number of test sequences to evaluate
        - time_step: Time interval for plotting
        - preprocess_inputs: Set to True if you want to preprocess raw inputs here
    """
    model.eval()

    fig, axs = plt.subplots(num_trajectories, 1, figsize=(10, 2.5 * num_trajectories), dpi=120)
    if num_trajectories == 1:
        axs = [axs]

    with torch.no_grad():
        for i in range(num_trajectories):
            input_seq = test_inputs[i]  # [T, 1]

            if preprocess_inputs:
                # OPTIONAL: Standardize input using time-step-specific mean/std
                input_seq = (input_seq.squeeze(-1).numpy() - input_mean) / input_std
                input_seq = torch.tensor(input_seq, dtype=torch.float32).unsqueeze(0).unsqueeze(-1)
            else:
                input_seq = input_seq.unsqueeze(0)  # [1, T, 1]

            true_output = test_outputs[i].squeeze(-1)  # [T]
            x0 = true_output[0].unsqueeze(0).unsqueeze(0)  # [1, 1]
            t = torch.linspace(0, 1, input_seq.size(1))  # [T]

            pred_output = model(x0, input_seq, t).squeeze(0).squeeze(-1)  # [T]

            # De-standardize predictions and true values
            pred_output_denorm = destandardize(pred_output.numpy(), output_mean, output_std)
            true_output_denorm = destandardize(true_output.numpy(), output_mean, output_std)

            time_axis = np.arange(len(pred_output_denorm)) * time_step

            ax = axs[i]
            ax.plot(time_axis, true_output_denorm, label="True Voltage", linewidth=2, linestyle="--")
            ax.plot(time_axis, pred_output_denorm, label="Predicted Voltage", linewidth=2)
            ax.set_title(f"Trajectory {i+1}")
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Voltage (V)")
            ax.grid(True)
            ax.legend()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    BASE_DIR = Path(__file__).resolve().parent.parent
    TRAIN_PATH = BASE_DIR / "data" / "training_data.pkl"
    TEST_PATH = BASE_DIR / "data" / "test_data.pkl"

    # Assume load_data properly defined to load and return processed inputs and outputs
    train_inputs, train_outputs, test_inputs, test_outputs, (input_mean, input_std), (output_mean, output_std) = load_data(TRAIN_PATH, TEST_PATH)
    train_loader = create_data_loader(train_inputs, train_outputs)

    ode_func = ODEFunc(state_dim=1, hidden_dim=128)
    neural_ode = NeuralODE(state_dim=1, hidden_dim=128)
    
    # Load model checkpoint
    model_path = Path("neural_ode_epoch_new.pth")
    if model_path.exists():
        print(f"Loading model from {model_path}")
        neural_ode.load_state_dict(torch.load(model_path))
        evaluate_model(neural_ode, test_inputs, test_outputs, input_mean, input_std, output_mean, output_std,
                    num_trajectories=5)
    else:
        print("Model checkpoint not found.")
        train_neural_ode(neural_ode, train_loader, num_epochs=300, learning_rate=0.001)
        print("Training complete.")


   


