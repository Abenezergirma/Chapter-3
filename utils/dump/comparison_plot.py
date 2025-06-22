import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize


plt.rcParams["text.usetex"] = True

# Constants
BASE_DIR = Path(__file__).resolve().parent
MODELS = ["voltage NODE", "voltage PINN"]
CATEGORIES = ["EM1", "EM3"] #["Short", "Long", "Infeasible"]
NUM_PREDICTED = 50  # Change this to >1 to trigger heatmap plotting
BATTERY_CALIBRATION = 2.0
TIME_SHIFT = 0
SHORT_OFFSET = -0.02 #0.318
SAVE_DIR = BASE_DIR / "results"
SAVE_DIR.mkdir(parents=True, exist_ok=True)

def load_trajectories(file_path):
    if not os.path.exists(file_path):
        print(f"Warning: {file_path} not found.")
        return []
    with open(file_path, "rb") as f:
        return pickle.load(f)

def plot_comparison(predicted_dict, actual_node, category):
    fig, ax = plt.subplots(figsize=(5, 4), dpi=300)

    # Plot actual trajectory from NODE model
    actual = actual_node[0]
    actual_time = np.array(actual["time"])
    actual_voltage = np.array(actual["voltage"]) - BATTERY_CALIBRATION
    ax.plot(actual_time, actual_voltage, linestyle="solid", linewidth=1.2, alpha=0.9, label="Actual", color="black")

    if NUM_PREDICTED > 1:
        # Shared colormap for both models
        cmap = plt.cm.viridis
        norm = Normalize(vmin=1, vmax=NUM_PREDICTED)
        colors = cmap(np.linspace(0, 1, NUM_PREDICTED))

        for model_name in ["voltage NODE", "voltage PINN"]:
            label = "NODE" if model_name == "voltage NODE" else "PINN"
            predicted_trajectories = predicted_dict[model_name]

            for idx, predicted in enumerate(predicted_trajectories[:NUM_PREDICTED]):
                pred_time = np.array(predicted["time"])
                pred_voltage = np.array(predicted["voltage"])

                if category == "Short":
                    pred_voltage += SHORT_OFFSET
                pred_voltage -= BATTERY_CALIBRATION

                ax.plot(pred_time + TIME_SHIFT, pred_voltage, linestyle="--", linewidth=1.0, alpha=0.9,
                        color=colors[idx], label=label if idx == 0 else None)

        # Add shared colorbar
        sm = ScalarMappable(cmap=cmap, norm=norm)
        cbar = fig.colorbar(sm, ax=ax, pad=0.02, aspect=30)
        cbar.set_label("Trajectory Index", fontsize=10)

    else:
        # Single line style for NODE and PINN
        style_map = {
            "voltage NODE": {"color": "blue", "label": "NODE"},
            "voltage PINN": {"color": "green", "label": "PINN"}
        }

        for model_name in ["voltage NODE", "voltage PINN"]:
            predicted = predicted_dict[model_name][0]
            pred_time = np.array(predicted["time"])
            pred_voltage = np.array(predicted["voltage"])

            if category == "Short":
                pred_voltage += SHORT_OFFSET
            pred_voltage -= BATTERY_CALIBRATION

            ax.plot(pred_time + TIME_SHIFT, pred_voltage, linestyle="--", linewidth=1.2, alpha=0.95,
                    color=style_map[model_name]["color"], label=style_map[model_name]["label"])

    # Voltage threshold line
    ax.axhline(y=18, color='red', linestyle='dashed', linewidth=1.0, label="Voltage Threshold")

    # Axis settings
    ax.set_xlabel("Time (s)", fontsize=12)
    ax.set_ylabel("Voltage (V)", fontsize=12)
    ax.grid(True, linestyle="--", alpha=0.5)

    # Legend
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), fontsize=10, frameon=False)

    # Save figure
    plt.tight_layout()
    save_path = SAVE_DIR / f"{category}_comparison.pdf"
    plt.savefig(save_path, format="pdf", bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")
    
def plot_combined_comparison(predicted_dict_all, actual_node_all):
    fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(6.5, 2.5), dpi=300)

    category_labels = ["Long", "Infeasible", "Short"]

    for i, category in enumerate(category_labels):
        ax = axs[i]
        predicted_dict = predicted_dict_all[category]
        actual_node = actual_node_all[category]

        # Plot actual trajectory
        actual = actual_node[0]
        actual_time = np.array(actual["time"])
        actual_voltage = np.array(actual["voltage"]) - BATTERY_CALIBRATION
        ax.plot(actual_time, actual_voltage, linestyle="solid", linewidth=1.2, alpha=0.9, color="black", label="Actual" if i == 0 else None)

        if NUM_PREDICTED > 1:
            cmap = plt.cm.viridis
            colors = cmap(np.linspace(0, 1, NUM_PREDICTED))

            for model_name in ["voltage NODE", "voltage PINN"]:
                label = "NODE" if model_name == "voltage NODE" else "PINN"
                predicted_trajectories = predicted_dict[model_name]

                for idx, predicted in enumerate(predicted_trajectories[:NUM_PREDICTED]):
                    pred_time = np.array(predicted["time"])
                    pred_voltage = np.array(predicted["voltage"])
                    if category == "Short":
                        pred_voltage += SHORT_OFFSET
                    pred_voltage -= BATTERY_CALIBRATION

                    ax.plot(pred_time + TIME_SHIFT, pred_voltage, linestyle="--", linewidth=1.0, alpha=0.9,
                            color=colors[idx], label=label if idx == 0 and i == 0 else None)
        else:
            style_map = {
                "voltage NODE": {"color": "blue", "label": "Neural-ODE"},
                "voltage PINN": {"color": "green", "label": "PINN"}
            }
            for model_name in ["voltage NODE", "voltage PINN"]:
                predicted = predicted_dict[model_name][0]
                pred_time = np.array(predicted["time"])
                pred_voltage = np.array(predicted["voltage"])
                if category == "Short":
                    pred_voltage += SHORT_OFFSET
                pred_voltage -= BATTERY_CALIBRATION

                ax.plot(pred_time + TIME_SHIFT, pred_voltage, linestyle="--", linewidth=1.2, alpha=0.95,
                        color=style_map[model_name]["color"], label=style_map[model_name]["label"] if i == 0 else None)

        ax.axhline(y=18, color='red', linestyle='dashed', linewidth=1.0)
        ax.set_xlabel("Time (s)", fontsize=10)
        if i == 0:
            ax.set_ylabel("Voltage (V)", fontsize=10)
            # Add legend inside the first plot (top-left)
            ax.legend(loc="upper right", fontsize=8, frameon=True)
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.set_title(rf"\textrm{{{category} Voltage Profile}}", fontsize=10)


    plt.tight_layout()
    combined_path = SAVE_DIR / "combined_comparison.pdf"
    plt.savefig(combined_path, format="pdf", bbox_inches="tight")
    plt.close()
    print(f"Saved combined figure: {combined_path}")

def plot_combined_comparison_cmap(predicted_dict_all, actual_node_all, model_mode="node", cmap_name="viridis"):
    """
    Plots a 3x1 combined voltage comparison figure across Long, Infeasible, Short trajectories.

    Args:
        predicted_dict_all: dict of predicted trajectories per category
        actual_node_all: dict of actual trajectories (from NODE)
        model_mode: 'compare', 'NODE', or 'PINN'
        cmap_name: colormap name (e.g., 'viridis', 'plasma')
    """

    fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(6.5, 2.5), dpi=300, constrained_layout=True)
    category_labels = ["Long", "Infeasible", "Short"]

    for i, category in enumerate(category_labels):
        ax = axs[i]
        predicted_dict = predicted_dict_all[category]
        actual_node = actual_node_all[category]

        # Plot actual trajectory from NODE
        actual = actual_node[0]
        actual_time = np.array(actual["time"])
        actual_voltage = np.array(actual["voltage"]) - BATTERY_CALIBRATION
        ax.plot(actual_time, actual_voltage, linestyle="solid", linewidth=1.2, alpha=0.9, color="black", label="Actual" if i == 0 else None)

        # Select models to plot
        model_mode_upper = model_mode.upper()
        if model_mode_upper == "COMPARE":
            models_to_plot = ["voltage NODE", "voltage PINN"]
        elif model_mode_upper in ["NODE", "PINN"]:
            models_to_plot = [f"voltage {model_mode_upper}"]
        else:
            raise ValueError(f"Invalid model_mode: {model_mode}. Use 'compare', 'NODE', or 'PINN'.")

        # Plot predicted
        if NUM_PREDICTED > 1:
            cmap = plt.get_cmap(cmap_name)
            colors = cmap(np.linspace(0, 1, NUM_PREDICTED))
            for model_name in models_to_plot:
                label = "Neural-ODE" if model_name == "voltage NODE" else "PINN"
                for idx, predicted in enumerate(predicted_dict[model_name][:NUM_PREDICTED]):
                    pred_time = np.array(predicted["time"])
                    pred_voltage = np.array(predicted["voltage"])
                    if category.lower() == "short":
                        pred_voltage += SHORT_OFFSET
                    pred_voltage -= BATTERY_CALIBRATION
                    ax.plot(pred_time, pred_voltage, linestyle="--", linewidth=1.0, alpha=0.9,
                            color=colors[idx], label=label if idx == 0 and i == 0 else None)
        else:
            style_map = {
                "voltage NODE": {"color": "blue", "label": "Neural-ODE"},
                "voltage PINN": {"color": "green", "label": "PINN"}
            }
            for model_name in models_to_plot:
                predicted = predicted_dict[model_name][0]
                pred_time = np.array(predicted["time"])
                pred_voltage = np.array(predicted["voltage"])
                if category.lower() == "short":
                    pred_voltage += SHORT_OFFSET
                pred_voltage -= BATTERY_CALIBRATION
                ax.plot(pred_time, pred_voltage, linestyle="--", linewidth=1.2, alpha=0.95,
                        color=style_map[model_name]["color"],
                        label=style_map[model_name]["label"] if i == 0 else None)

        # Format axes
        ax.axhline(y=18, color='red', linestyle='dashed', linewidth=1.0)
        ax.set_xlabel("Time (s)", fontsize=10)
        if i == 0:
            ax.set_ylabel("Voltage (V)", fontsize=10)
            ax.legend(loc="upper right", fontsize=8, frameon=True)
        ax.set_title(rf"\textrm{{{category} Voltage Profile}}", fontsize=10)
        ax.grid(True, linestyle="--", alpha=0.5)

    # Shared colorbar if needed
    if NUM_PREDICTED > 1:
        norm = Normalize(vmin=1, vmax=NUM_PREDICTED)
        sm = ScalarMappable(cmap=plt.get_cmap(cmap_name), norm=norm)
        cbar = fig.colorbar(sm, ax=axs, orientation="vertical", pad=0.02, aspect=30)
        cbar.set_label("Trajectory Index", fontsize=10)

    # Save figure
    save_path = f"combined_voltage_comparison_{model_mode.lower()}_{cmap_name}.pdf"
    plt.savefig(save_path, format="pdf", bbox_inches="tight")
    plt.close()
    print(f"Saved combined figure: {save_path}")


def plot_comparison_cmap_EMs(predicted_dict_all, actual_node_all, model_mode="node", cmap_name="viridis"):
    """
    Plots a 2x1 voltage comparison figure for EM1 and EM2 trajectories.

    Args:
        predicted_dict_all: dict of predicted trajectories per category
        actual_node_all: dict of actual trajectories (from NODE)
        model_mode: 'compare', 'NODE', or 'PINN'
        cmap_name: colormap name (e.g., 'viridis', 'plasma')
    """

    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(5, 2.2), dpi=300, constrained_layout=True)
    category_labels = ["EM1", "EM3"]

    for i, category in enumerate(category_labels):
        ax = axs[i]
        predicted_dict = predicted_dict_all[category]
        actual_node = actual_node_all[category]

        # Plot actual trajectory
        actual = actual_node[0]
        actual_time = np.array(actual["time"])
        actual_voltage = np.array(actual["voltage"]) - BATTERY_CALIBRATION
        ax.plot(actual_time, actual_voltage, linestyle="solid", linewidth=1.2, alpha=0.9, color="black", label="Actual" if i == 0 else None)

        # Decide what to plot
        model_mode_upper = model_mode.upper()
        if model_mode_upper == "COMPARE":
            models_to_plot = ["voltage NODE", "voltage PINN"]
        elif model_mode_upper in ["NODE", "PINN"]:
            models_to_plot = [f"voltage {model_mode_upper}"]
        else:
            raise ValueError(f"Invalid model_mode: {model_mode}. Use 'compare', 'NODE', or 'PINN'.")

        # Plot predicted
        if NUM_PREDICTED > 1:
            cmap = plt.get_cmap(cmap_name)
            colors = cmap(np.linspace(0, 1, NUM_PREDICTED))
            for model_name in models_to_plot:
                label = "Neural-ODE" if model_name == "voltage NODE" else "PINN"
                for idx, predicted in enumerate(predicted_dict[model_name][:NUM_PREDICTED]):
                    pred_time = np.array(predicted["time"])
                    pred_voltage = np.array(predicted["voltage"]) - BATTERY_CALIBRATION
                    ax.plot(pred_time, pred_voltage, linestyle="--", linewidth=1.0, alpha=0.9,
                            color=colors[idx], label=label if idx == 0 and i == 0 else None)
        else:
            style_map = {
                "voltage NODE": {"color": "blue", "label": "Neural-ODE"},
                "voltage PINN": {"color": "green", "label": "PINN"}
            }
            for model_name in models_to_plot:
                predicted = predicted_dict[model_name][0]
                pred_time = np.array(predicted["time"])
                pred_voltage = np.array(predicted["voltage"]) - BATTERY_CALIBRATION
                ax.plot(pred_time, pred_voltage, linestyle="--", linewidth=1.2, alpha=0.95,
                        color=style_map[model_name]["color"],
                        label=style_map[model_name]["label"] if i == 0 else None)

        # Style
        ax.axhline(y=18, color='red', linestyle='dashed', linewidth=1.0)
        ax.set_xlabel("Time (s)", fontsize=10)
        if i == 0:
            ax.set_ylabel("Voltage (V)", fontsize=10)
            ax.legend(loc="upper right", fontsize=8, frameon=True)
        ax.set_title(rf"\textrm{{{category} Voltage Profile}}", fontsize=10)
        ax.grid(True, linestyle="--", alpha=0.5)

    # Shared colorbar
    if NUM_PREDICTED > 1:
        norm = Normalize(vmin=1, vmax=NUM_PREDICTED)
        sm = ScalarMappable(cmap=plt.get_cmap(cmap_name), norm=norm)
        cbar = fig.colorbar(sm, ax=axs, orientation="vertical", pad=0.02, aspect=30)
        cbar.set_label("Trajectory Index", fontsize=10)

    # Save
    save_path = f"combined_voltage_comparison_EM_{model_mode.lower()}_{cmap_name}.pdf"
    plt.savefig(save_path, format="pdf", bbox_inches="tight")
    plt.close()
    print(f"✅ Saved EM1/EM2 figure: {save_path}")

# Collect data for combined plot
predicted_all = {}
actual_all = {}

for category in CATEGORIES:
    predicted_trajectories = {}
    actual_trajectory_node = []

    for model in MODELS:
        model_dir = BASE_DIR / "new_voltages"/ model / category
        predicted_path = model_dir / "saved_predicted_voltage_trajectories.pkl"
        actual_path = model_dir / "saved_actual_voltage_trajectories.pkl"

        predicted = load_trajectories(predicted_path)
        predicted_trajectories[model] = predicted

        if model == "voltage NODE":
            actual_trajectory_node = load_trajectories(actual_path)

    predicted_all[category] = predicted_trajectories
    actual_all[category] = actual_trajectory_node

    plot_comparison(predicted_trajectories, actual_trajectory_node, category)

# Generate final combined figure
plot_comparison_cmap_EMs(predicted_all, actual_all)

