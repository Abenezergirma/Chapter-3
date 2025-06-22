# 🔋 Data-Driven Battery Modeling with Neural ODE and PINN

This project implements two data-driven modeling approaches—**Neural ODE** and **Physics-Informed Neural Networks (PINN)**—to predict the voltage response of a Tarot T18 drone battery given a current or power profile. The project integrates physics-based constraints and benchmarking with an electrochemical battery model.

---

## 📁 Project Structure

```
data_driven_battery_modeling/
│
├── battery_model/                     # Physics-based electrochemical battery models
│   └── battery_electrochem_TarotT18.py
│
├── data/                              # All data and simulation results
│   ├── training_data.pkl
│   ├── test_data.pkl
│   ├── SimulationResults/
│   │   ├── projected_profiles_long.mat
│   │   └── saved_current_profiles_infeasible.pkl
│   └── ...
│
├── prediction/                        # Scripts for evaluating and visualizing predictions
│   ├── predict_neural_ode_voltage.py
│   └── predict_pinn_voltage.py
│
├── scripts/                           # Dataset generation and preprocessing utilities
│   ├── generate_dataset.py
│   └── preprocess_power_profile.py
│
├── training/                          # Model training scripts
│   ├── train_neural_ode.py
│   └── train_pinn.py
│
├── saved_models/                      # Trained model checkpoints
│   ├── neural_ode/
│   │   └── model_checkpoints/
│   │       └── neural_ode_epoch*.pth
│   └── pinn_model_best.pth
│
├── README.md
└── requirements.txt (optional)
```

---

## 🧠 Models

### 1. Neural ODE
- **Training**: `training/train_neural_ode.py`
- **Prediction**: `prediction/predict_neural_ode_voltage.py`
- **Model Checkpoints**: Saved in `saved_models/neural_ode/model_checkpoints/`

### 2. PINN (Physics-Informed Neural Network)
- **Training**: `training/train_pinn.py`
- **Prediction**: `prediction/predict_pinn_voltage.py`
- **Model Checkpoint**: `saved_models/pinn_model_best.pth`

---

## ⚙️ Dataset Generation

- **Script**: `scripts/generate_dataset.py`
- **Output**: 
  - `data/training_data.pkl`
  - `data/test_data.pkl`

Includes full-flight and mid-flight current profiles simulated with the physics-based battery model.

---

## ⚡ Power Profile Preprocessing

- **Script**: `scripts/preprocess_power_profile.py`
- Converts power profiles (from .mat files) to current profiles using a simplified Rint battery model.
- Saves result to: `data/SimulationResults/saved_current_profiles_infeasible.pkl`

---

## 📊 Prediction Comparison

Each prediction script (Neural ODE and PINN) compares the predicted battery voltage against the simulated ground truth using the electrochemical battery model.

Outputs include:
- Visual plots of voltage over time.
- RMSE and MAE error metrics.
- Saved figures (`Voltage_Comparison.pdf`, etc.)

---

## 🧪 How to Run (from project root)

> ⚠️ Important: You must add the project root to `PYTHONPATH` or do it manually in the script via `sys.path`.

### Generate Training Dataset
```bash
python scripts/generate_dataset.py
```

### Preprocess Power Profile
```bash
python scripts/preprocess_power_profile.py
```

### Train Neural ODE
```bash
python training/train_neural_ode.py
```

### Predict with Neural ODE
```bash
python prediction/predict_neural_ode_voltage.py
```

### Train PINN
```bash
python training/train_pinn.py
```

### Predict with PINN
```bash
python prediction/predict_pinn_voltage.py
```

---

## 📌 Notes

- The reference power profile `EnergyReq.mat` must be placed in the project root or data folder.
- All voltage predictions are normalized during training and denormalized for evaluation using min/max values computed from training data.
- Physics constraints (Nernst, BV, RC) are included in the PINN loss function for higher accuracy and interpretability.

---

## 🔧 Troubleshooting

- If you see `ModuleNotFoundError`, make sure to add the project root to `sys.path` in your scripts or launch scripts with:
  ```bash
  PYTHONPATH=. python prediction/predict_neural_ode_voltage.py
  ```


