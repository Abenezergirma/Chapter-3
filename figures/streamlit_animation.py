import sys
from types import ModuleType
import matplotlib
matplotlib.use('Agg')
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pickle
from pathlib import Path
import time
import pandas as pd
from io import BytesIO

# Mock the plotly_chart module to avoid import errors
class MockModule(ModuleType):
    def __getattr__(self, name):
        return None

sys.modules['streamlit.elements.plotly_chart'] = MockModule('streamlit.elements.plotly_chart')

# --- Helper Functions ---

def load_saved_profiles(file_path):
    if not file_path.exists():
        st.error(f"File not found: {file_path}")
        return []
    with open(file_path, "rb") as f:
        return pickle.load(f)

def calculate_mae(pred_voltage, actual_voltage, actual_time, pred_time):
    """Calculate Mean Absolute Error between predicted and actual voltages."""
    pred_voltage_interp = np.interp(actual_time, pred_time, pred_voltage)
    return np.mean(np.abs(pred_voltage_interp - actual_voltage))

def plot_trajectory(pred, actual, trajectory_idx, colors, min_time, max_time, min_voltage, max_voltage):
    """Plot a single trajectory and return the figure."""
    plt.clf()
    fig, ax = plt.subplots(figsize=(6, 4), dpi=150)
    plt.style.use('seaborn-v0_8-whitegrid')

    # Plot lines
    pred_time = np.array(pred["time"])
    pred_voltage = np.array(pred["voltage"]) - 2
    actual_time = np.array(actual["time"])
    actual_voltage = np.array(actual["voltage"]) - 2

    if len(pred_time) == 0 or len(actual_time) == 0:
        return None

    pred_voltage_interp = np.interp(actual_time, pred_time, pred_voltage)
    ax.plot(actual_time, pred_voltage_interp, linestyle="--", lw=2, label="Neural-ODE Prediction", color=colors[trajectory_idx], zorder=3)
    ax.plot(actual_time, actual_voltage, linestyle="-", lw=2, label="Model-Based Actual", color=colors[trajectory_idx], zorder=3)
    ax.axhline(y=18.5, color='#FF4C4C', linestyle='--', lw=2, alpha=0.8, label="Voltage Threshold")
    ax.text(0.02, 0.95, f"Trajectory {trajectory_idx + 1}", transform=ax.transAxes, fontsize=10, va='top', ha='left')

    # Set plot limits and labels
    ax.set_xlim(min_time, max_time)
    ax.set_ylim(min_voltage, max_voltage)
    ax.set_xlabel("Time (s)", fontsize=12)
    ax.set_ylabel("Voltage (V)", fontsize=12)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="upper right")

    return fig

def get_data_table(pred, actual):
    """Create a DataFrame for the trajectory data."""
    pred_time = np.array(pred["time"])
    pred_voltage = np.array(pred["voltage"]) - 2
    actual_time = np.array(actual["time"])
    actual_voltage = np.array(actual["voltage"]) - 2
    pred_voltage_interp = np.interp(actual_time, pred_time, pred_voltage)
    return pd.DataFrame({
        "Time (s)": actual_time,
        "Actual Voltage (V)": actual_voltage,
        "Predicted Voltage (V)": pred_voltage_interp
    })

def save_plot_to_buffer(fig):
    """Save the figure to a BytesIO buffer as PNG."""
    buf = BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    return buf

# --- Streamlit App Starts ---

st.set_page_config(page_title="UAV Battery Voltage Prediction Dashboard", layout="wide")

# Custom CSS for styling
st.markdown("""
    <style>
    .main { background-color: #f5f5f5; }
    .stButton>button { background-color: #4CAF50; color: white; }
    .stSlider { background-color: #e0e0e0; padding: 10px; border-radius: 5px; }
    h1 { color: #2c3e50; }
    </style>
""", unsafe_allow_html=True)

st.title("🔋 UAV Battery Voltage Prediction Dashboard")
st.markdown("Explore Neural-ODE predicted vs. model-based actual voltage measurements across UAV mission profiles.")

# Load Data
BASE_DIR = Path(__file__).resolve().parents[1]
predicted_voltage_path = BASE_DIR / "voltage NODE" / "saved_predicted_voltage_trajectories.pkl"
actual_voltage_path = BASE_DIR  / "voltage NODE"/ "saved_actual_voltage_trajectories.pkl"

predicted_voltage_trajectories = load_saved_profiles(predicted_voltage_path)
actual_voltage_trajectories = load_saved_profiles(actual_voltage_path)

if not predicted_voltage_trajectories or not actual_voltage_trajectories:
    st.warning("Please check that your .pkl files are available in the app directory.")
else:
    num_trajectories = min(30, len(predicted_voltage_trajectories))
    colors = plt.cm.viridis(np.linspace(0, 1, num_trajectories))

    # Find dynamic plot limits
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

    # Create tabs for different views
    tab1, tab2, tab3 = st.tabs(["Animation", "Data Explorer", "Insights"])

    with tab1:
        st.header("Animation")
        st.markdown("Watch the voltage trajectories animate over time.")
        animate = st.checkbox("Run Animation", value=True)
        plot_placeholder = st.empty()
        progress = st.progress(0)

        if animate:
            for i in range(num_trajectories):
                fig = plot_trajectory(predicted_voltage_trajectories[i], actual_voltage_trajectories[i], i, colors, min_time, max_time, min_voltage, max_voltage)
                if fig:
                    with plot_placeholder.container():
                        st.pyplot(fig)
                    progress.progress((i + 1) / num_trajectories)
                    time.sleep(0.5)
                    plt.close(fig)

    with tab2:
        st.header("Data Explorer")
        trajectory_idx = st.slider("Select Trajectory", 0, num_trajectories - 1, 0, key="data_slider")
        fig = plot_trajectory(predicted_voltage_trajectories[trajectory_idx], actual_voltage_trajectories[trajectory_idx], trajectory_idx, colors, min_time, max_time, min_voltage, max_voltage)
        if fig:
            st.pyplot(fig)
            # Download button for the plot
            buf = save_plot_to_buffer(fig)
            st.download_button(
                label="Download Plot as PNG",
                data=buf,
                file_name=f"trajectory_{trajectory_idx + 1}.png",
                mime="image/png"
            )
            plt.close(fig)

        # Display data table
        st.subheader("Voltage Data")
        table = get_data_table(predicted_voltage_trajectories[trajectory_idx], actual_voltage_trajectories[trajectory_idx])
        st.dataframe(table)

    with tab3:
        st.header("Insights")
        st.markdown("Analyze the performance of the Neural-ODE model.")
        trajectory_idx = st.selectbox("Select Trajectory for Insights", range(num_trajectories), format_func=lambda x: f"Trajectory {x + 1}")
        pred = predicted_voltage_trajectories[trajectory_idx]
        actual = actual_voltage_trajectories[trajectory_idx]
        mae = calculate_mae(np.array(pred["voltage"]) - 2, np.array(actual["voltage"]) - 2, np.array(actual["time"]), np.array(pred["time"]))
        st.metric("Mean Absolute Error (MAE)", f"{mae:.4f} V")
        st.markdown(f"The MAE of {mae:.4f} V indicates the average difference between predicted and actual voltages for Trajectory {trajectory_idx + 1}.")