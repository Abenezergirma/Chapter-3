import matplotlib.pyplot as plt
import numpy as np
plt.rcParams["text.usetex"] = True

# Data
trajectories = ['Long', 'Infeasible', 'Short']
x = np.arange(len(trajectories))
width = 0.18

# Accuracy comparison
pinn_rmse = [0.86, 0.48, 0.39]
pinn_mae = [0.78, 0.40, 0.33]
ode_rmse = [0.14, 0.12, 0.13]
ode_mae = [0.11, 0.08, 0.07]

# Time comparison
ode_time = [4.50, 3.05, 2.30]
pinn_time = [0.76, 0.54, 0.75]
model_based_time = [126.92, 88.66, 63.10]

# ----- Accuracy Figure -----
fig1, ax1 = plt.subplots(figsize=(6, 4))
ax1.bar(x - 1.5*width, pinn_rmse, width, label='PINN RMSE')
ax1.bar(x - 0.5*width, pinn_mae, width, label='PINN MAE')
ax1.bar(x + 0.5*width, ode_rmse, width, label='Neural-ODE RMSE')
ax1.bar(x + 1.5*width, ode_mae, width, label='Neural-ODE MAE')

ax1.set_ylabel('Voltage Error (V)', fontsize=12)
ax1.set_xlabel('Trajectory Type', fontsize=12)
ax1.set_title('Prediction Accuracy Comparison', fontsize=14)
ax1.set_xticks(x)
ax1.set_xticklabels(trajectories, fontsize=11)
ax1.grid(axis='y', linestyle='--', alpha=0.7)
ax1.legend(fontsize=10)
fig1.tight_layout()
fig1.savefig('accuracy_comparison.pdf', format='pdf', bbox_inches='tight')

# ----- Computation Time Figure -----
fig2, ax2 = plt.subplots(figsize=(6, 4))
ax2.bar(x - width, pinn_time, width, label='PINN')
ax2.bar(x, ode_time, width, label='Neural-ODE')
ax2.bar(x + width, model_based_time, width, label='Model-Based')

ax2.set_ylabel('Computation Time (sec)', fontsize=12)
ax2.set_xlabel('Trajectory Type', fontsize=12)
ax2.set_title('Computation Time Comparison', fontsize=14)
ax2.set_xticks(x)
ax2.set_xticklabels(trajectories, fontsize=11)
ax2.grid(axis='y', linestyle='--', alpha=0.7)
ax2.legend(fontsize=10)
fig2.tight_layout()
fig2.savefig('computation_time_comparison.pdf', format='pdf', bbox_inches='tight')

# ----- Combined 2-Column Layout -----
def plot_combined_accuracy_and_time():
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(6.5, 2.8), dpi=300)

    # --- Left: Accuracy ---
    axs[0].bar(x - 1.5*width, pinn_rmse, width, label='PINN RMSE')
    axs[0].bar(x - 0.5*width, pinn_mae, width, label='PINN MAE')
    axs[0].bar(x + 0.5*width, ode_rmse, width, label='Neural-ODE RMSE')
    axs[0].bar(x + 1.5*width, ode_mae, width, label='Neural-ODE MAE')
    axs[0].set_ylabel('Voltage Error (V)', fontsize=11)
    axs[0].set_xlabel('Trajectory Type', fontsize=10)
    axs[0].set_xticks(x)
    axs[0].set_xticklabels(trajectories, fontsize=10)
    axs[0].grid(axis='y', linestyle='--', alpha=0.7)
    axs[0].legend(fontsize=9, frameon=False)

    # --- Right: Time ---
    axs[1].bar(x - width, pinn_time, width, label='PINN')
    axs[1].bar(x, ode_time, width, label='Neural-ODE')
    axs[1].bar(x + width, model_based_time, width, label='Model-Based')
    axs[1].set_ylabel('Computation Time (sec)', fontsize=11)
    axs[1].set_xlabel('Trajectory Type', fontsize=10)
    axs[1].set_xticks(x)
    axs[1].set_xticklabels(trajectories, fontsize=10)
    axs[1].grid(axis='y', linestyle='--', alpha=0.7)
    axs[1].legend(fontsize=9, frameon=False)

    # Layout & Save
    plt.tight_layout()
    fig.savefig('comparison_combined.pdf', format='pdf', bbox_inches='tight')
    plt.close()
    print("Saved combined comparison figure: comparison_combined.pdf")

# Call the combined plot function
plot_combined_accuracy_and_time()

# Optionally show the figures if you're running locally
# plt.show()
