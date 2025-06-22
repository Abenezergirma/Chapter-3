import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
import numpy as np
import os
import pickle
from pathlib import Path
plt.rcParams["text.usetex"] = True


def load_saved_current_profiles(file_path):
    if not os.path.exists(file_path):
        print(f"Error: Saved current profiles file {file_path} not found.")
        return []
    with open(file_path, "rb") as f:
        return pickle.load(f)


def animate_voltage_trajectories(predicted_voltage_trajectories, actual_voltage_trajectories, save_path="animated_voltage_prediction.gif"):
    # Set up figure with presentation style
    plt.style.use('seaborn-v0_8-whitegrid')  # Modern, clean style
    fig, ax = plt.subplots(figsize=(6, 4), dpi=300)
    # print(predicted_voltage_trajectories)
    num_trajectories = min(55, len(predicted_voltage_trajectories))
    colors = plt.cm.viridis(np.linspace(0, 1, num_trajectories))

    # Initialize plot elements
    time_line_pred, = ax.plot([], [], linestyle="--", lw=2.5, label="LSTM", zorder=3)
    time_line_actual, = ax.plot([], [], linestyle="-", lw=2.5, label="Model Based", zorder=3)
    threshold_line = ax.axhline(y=18.5, color='#FF4C4C', linestyle='--', lw=2, alpha=0.8, label="Threshold")
    # text_label = ax.text(0.5, 0.98, "", transform=ax.transAxes, fontsize=12, va='top', ha='left', bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))
    text_label = ax.text(0.02, 0.98, "", transform=ax.transAxes, fontsize=12, va='top', ha='left', bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))
    # Background traces for first trajectory
    first_trace_pred, = ax.plot([], [], linestyle="--", lw=1.5, color='gray', alpha=0.2, zorder=1)
    first_trace_actual, = ax.plot([], [], linestyle="-", lw=1.5, color='gray', alpha=0.2, zorder=1)

    # Dynamically calculate limits
    all_times = []
    all_voltages = []
    for actual in actual_voltage_trajectories[:num_trajectories]:
        if "time" in actual and "voltage" in actual:
            all_times.extend(actual["time"])
            all_voltages.extend(np.array(actual["voltage"]) - 2)
    min_time = min(all_times) if all_times else 0
    max_time = max(all_times) if all_times else 300
    min_voltage = min(all_voltages) if all_voltages else 15
    max_voltage = max(all_voltages) if all_voltages else 26
    time_padding = 0.05 * (max_time - min_time)
    voltage_padding = 0.1 * (max_voltage - min_voltage)
    ax.set_xlim(min_time - time_padding, max_time + time_padding)
    ax.set_ylim(min_voltage - voltage_padding, max_voltage + voltage_padding)

    # Customize plot aesthetics
    ax.set_xlabel("Time (s)", fontsize=14, weight='bold')
    ax.set_ylabel("Voltage (V)", fontsize=14, weight='bold')
    # ax.set_title("Battery Voltage Prediction: Model vs. Actual", fontsize=16, weight='bold', pad=15)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)

    # Add colorbar
    sm = ScalarMappable(cmap=plt.cm.viridis, norm=Normalize(vmin=1, vmax=num_trajectories))
    cbar = fig.colorbar(sm, ax=ax, aspect=30, pad=0.02)
    cbar.set_label("Trajectory Index", fontsize=12)
    cbar.ax.tick_params(labelsize=10)

    # Add legend with proxy artists for consistent colors
    proxy_neural_ode = Line2D([0], [0], linestyle="--", lw=2.5, color=colors[num_trajectories//2], label="LSTM")
    proxy_model_based = Line2D([0], [0], linestyle="-", lw=2.5, color=colors[num_trajectories//2], label="Model Based")
    ax.legend(handles=[proxy_neural_ode, proxy_model_based, threshold_line], 
             loc="upper right", fontsize=10, frameon=True, facecolor='white', framealpha=0.7, edgecolor='none')

    # Store first trajectory data for background traces
    first_pred_time, first_pred_voltage = None, None
    first_actual_time, first_actual_voltage = None, None

    def init():
        time_line_pred.set_data([], [])
        time_line_actual.set_data([], [])
        first_trace_pred.set_data([], [])
        first_trace_actual.set_data([], [])
        text_label.set_text("")
        return time_line_pred, time_line_actual, first_trace_pred, first_trace_actual, text_label

    def update(i):
        pred = predicted_voltage_trajectories[i]
        actual = actual_voltage_trajectories[i]
        pred_time, pred_voltage = np.array(pred["time"]), np.array(pred["voltage"]) -2
        actual_time, actual_voltage = np.array(actual["time"]), np.array(actual["voltage"]) -2

        if pred_time.size == 0 or actual_time.size == 0:
            text_label.set_text(f"Prediction Time: 0.0 s\n[Empty Data]")
            return time_line_pred, time_line_actual, first_trace_pred, first_trace_actual, text_label

        try:
            pred_voltage_interp = np.interp(actual_time, pred_time, pred_voltage)
        except Exception as e:
            text_label.set_text(f"Prediction Time: {actual_time[0]:.1f} s\n[Interpolation Error: {e}]")
            return time_line_pred, time_line_actual, first_trace_pred, first_trace_actual, text_label

        # Update current trajectory
        time_line_pred.set_data(actual_time, pred_voltage_interp)
        time_line_actual.set_data(actual_time, actual_voltage)
        time_line_pred.set_color(colors[i])
        time_line_actual.set_color(colors[i])

        # Update first trajectory traces (after first frame)
        nonlocal first_pred_time, first_pred_voltage, first_actual_time, first_actual_voltage
        if i == 0:
            first_pred_time, first_pred_voltage = actual_time, pred_voltage_interp
            first_actual_time, first_actual_voltage = actual_time, actual_voltage
        if i > 0 and first_pred_time is not None:
            first_trace_pred.set_data(first_pred_time, first_pred_voltage)
            first_trace_actual.set_data(first_actual_time, first_actual_voltage)

        # Update text label with Prediction Time only
        initial_time = actual_time[0] if len(actual_time) > 0 else 0
        text_label.set_text(f"Prediction Time: {initial_time:.1f} s")

        # Dynamic subtitle
        ax.set_title(f"LSTM Battery Voltage Prediction {i+1}", fontsize=14, weight='bold', pad=15)

        return time_line_pred, time_line_actual, first_trace_pred, first_trace_actual, text_label

    # Create animation (faster, 200 ms per frame = 5 fps)
    anim = FuncAnimation(fig, update, frames=num_trajectories, init_func=init,
                         blit=True, interval=200, repeat=True)

    # Save as high-quality GIF
    anim.save(save_path, writer=PillowWriter(fps=5), dpi=300)
    plt.close()


if __name__ == "__main__":
    BASE_DIR = Path(__file__).resolve().parents[1]
    predicted_voltage_trajectories = load_saved_current_profiles(BASE_DIR / "voltage LSTM"/ "saved_predicted_voltage_trajectories.pkl")
    actual_voltage_trajectories = load_saved_current_profiles(BASE_DIR / "voltage LSTM"/"saved_actual_voltage_trajectories.pkl")
    animate_voltage_trajectories(predicted_voltage_trajectories, actual_voltage_trajectories)