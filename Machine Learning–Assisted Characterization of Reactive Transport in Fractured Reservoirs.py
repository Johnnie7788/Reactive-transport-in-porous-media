#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# -------------------------------------------------------------
# PROJECT:
# Machine Learning–Assisted Characterization of Reactive Transport
# in Fractured Reservoirs
# -------------------------------------------------------------

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error

# -----------------------------
# To write/read in the current folder:
project_dir = Path(".")
# If you want a specific Windows folder, uncomment and update:
# project_dir = Path(r"C:\Users\15107\Downloads\... \PROJECT")

csv_path      = project_dir / "fracture_dataset_project3.csv"
predictions_path = project_dir / "fracture_predictions_project3.csv"
fi_png        = project_dir / "feature_importance_project3.png"
parity_png    = project_dir / "parity_plots_project3.png"


# -----------------------------
rng = np.random.default_rng(2025)
N = 1500

# Fracture geometry & network
b_um      = rng.lognormal(mean=np.log(80.0), sigma=0.5,  size=N)   # aperture [µm]
spacing_m = rng.lognormal(mean=np.log(1.5), sigma=0.5,   size=N)   # spacing [m]
length_m  = rng.lognormal(mean=np.log(30.0), sigma=0.6,  size=N)   # length [m]
dens_1pm3 = rng.lognormal(mean=np.log(2.0), sigma=0.6,   size=N)   # volumetric density [1/m^3]
roughness = rng.uniform(0.1, 0.9, size=N)                          # 0..1

# Matrix & fluid
phi_matrix   = rng.uniform(0.08, 0.22, size=N)                     # porosity [-]
k_matrix_mD  = np.clip(rng.lognormal(np.log(2.0), 0.8, size=N), 0.01, 500.0)  # mD
salinity_gL  = rng.uniform(10.0, 200.0, size=N)                    # g/L
mu_cP        = rng.uniform(0.6, 1.6, size=N)                       # cP
T_C          = rng.uniform(25.0, 90.0, size=N)                     # °C
dPdx_Pa_m    = -rng.uniform(5e3, 5e4, size=N)                      # Pa/m (magnitude drives flow)
L_m          = rng.uniform(20.0, 200.0, size=N)                    # control length [m]

# Unit conversions
b_m          = b_um * 1e-6
mu_Pa_s      = mu_cP * 1e-3
k_matrix_m2  = k_matrix_mD * 9.869233e-16   # 1 mD = 9.869233e-16 m^2

# Physics-informed targets
# Fracture contribution (cubic law proxy + density → areal density)
areal_density = np.clip(dens_1pm3 * spacing_m, 1e-8, None)   # ~ fractures per m^2
k_fracture    = (b_m**2) / 12.0 * areal_density              # m^2 (proxy)
k_eff_m2      = k_matrix_m2 + k_fracture                      # simple parallel mix

# Darcy velocity: u = (k_eff/mu) * |dP/dx| / phi
u_m_per_s     = (k_eff_m2 / mu_Pa_s) * np.abs(dPdx_Pa_m) / np.clip(phi_matrix, 1e-3, None)

# Longitudinal dispersivity (empirical): grows with roughness, aperture, and decreases with spacing
alpha_L_m     = 0.05 + 0.30*roughness + 2.0*b_m + 0.5/np.clip(spacing_m, 0.05, None)

# Molecular diffusion (saltwater, crude T & salinity dependence)
Dm            = (1.0e-9 + (T_C - 25.0)*1.5e-11) * (1.0 - 0.002*salinity_gL)
Dm            = np.clip(Dm, 3e-10, 4e-9)
# Hydrodynamic dispersion
D_m2_per_s    = Dm + alpha_L_m * u_m_per_s

# First-order reactive decay (Arrhenius-like with salinity factor)
k0 = 1e-6
k_react_s_inv = k0 * np.exp(0.04*(T_C - 25.0)) * (1.0 + 0.002*salinity_gL)

# Mass removal over control length (plug-flow): 1 − exp(−k * L / u)
removal_frac  = 1.0 - np.exp(-k_react_s_inv * L_m / np.clip(u_m_per_s, 1e-12, None))
removal_frac  = np.clip(removal_frac, 0.0, 1.0)

# Assemble and SAVE dataset
dataset = pd.DataFrame({
    "b_um": b_um,
    "spacing_m": spacing_m,
    "length_m": length_m,
    "fracture_density_1pm3": dens_1pm3,
    "roughness": roughness,
    "phi_matrix": phi_matrix,
    "k_matrix_mD": k_matrix_mD,
    "salinity_gL": salinity_gL,
    "mu_cP": mu_cP,
    "T_C": T_C,
    "dPdx_Pa_m": dPdx_Pa_m,
    "L_m": L_m,
    # Derived quantities (optional features for downstream analysis)
    "alpha_L_m": alpha_L_m,
    "u_m_per_s": u_m_per_s,
    "D_m2_per_s": D_m2_per_s,
    # TARGETS for ML
    "k_eff_m2": k_eff_m2,
    "removal_frac": removal_frac
})
dataset.to_csv(csv_path, index=False)
print(f"[OK] Wrote dataset to: {csv_path}")

# -----------------------------
# 2) READ the CSV (explicit line)
# -----------------------------
df = pd.read_csv(csv_path)   # <-- required explicit CSV read

# -----------------------------
# 3) ML setup
# -----------------------------
feature_cols = [
    "b_um","spacing_m","length_m","fracture_density_1pm3","roughness",
    "phi_matrix","k_matrix_mD","salinity_gL","mu_cP","T_C","dPdx_Pa_m","L_m"
]
X  = df[feature_cols].values
yK = df["k_eff_m2"].values
yR = df["removal_frac"].values

# Single consistent split for both targets
X_tr, X_te, yK_tr, yK_te, yR_tr, yR_te = train_test_split(
    X, yK, yR, test_size=0.2, random_state=42
)

# Models
rf_k = RandomForestRegressor(n_estimators=400, random_state=42, n_jobs=-1)
rf_r = RandomForestRegressor(n_estimators=400, random_state=42, n_jobs=-1)

rf_k.fit(X_tr, yK_tr)
rf_r.fit(X_tr, yR_tr)

yK_pred = rf_k.predict(X_te)
yR_pred = rf_r.predict(X_te)

# Metrics
k_r2  = r2_score(yK_te, yK_pred)
k_mae = mean_absolute_error(yK_te, yK_pred)
r_r2  = r2_score(yR_te, yR_pred)
r_mae = mean_absolute_error(yR_te, yR_pred)

print(f"k_eff  → R²={k_r2:.3f}, MAE={k_mae:.3e} (m²)")
print(f"removal→ R²={r_r2:.3f}, MAE={r_mae:.4f} (-)")

# -----------------------------
# 4) Save predictions
# -----------------------------
preds = pd.DataFrame({
    "k_eff_true": yK_te, "k_eff_pred": yK_pred,
    "removal_true": yR_te, "removal_pred": yR_pred
})
preds.to_csv(predictions_path, index=False)
print(f"[OK] Wrote predictions to: {predictions_path}")

# -----------------------------
# 5) Feature importance + parity plots
# -----------------------------
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

# (a) Feature importance for removal model
imp = rf_r.feature_importances_
order = np.argsort(imp)[::-1]
axes[0].bar(range(len(feature_cols)), imp[order])
axes[0].set_xticks(range(len(feature_cols)))
axes[0].set_xticklabels(np.array(feature_cols)[order], rotation=40, ha="right")
axes[0].set_ylabel("Importance")
axes[0].set_title("Feature Importance (removal_frac)")

# (b) Parity for k_eff
axes[1].scatter(yK_te, yK_pred, s=12)
mink, maxk = float(np.min(yK_te)), float(np.max(yK_te))
axes[1].plot([mink, maxk], [mink, maxk])
axes[1].set_xlabel("k_eff true (m²)")
axes[1].set_ylabel("k_eff pred (m²)")
axes[1].set_title(f"k_eff parity (R²={k_r2:.2f})")

# (c) Parity for removal
axes[2].scatter(yR_te, yR_pred, s=12)
axes[2].plot([0,1],[0,1])
axes[2].set_xlim(0,1); axes[2].set_ylim(0,1)
axes[2].set_xlabel("removal true (-)")
axes[2].set_ylabel("removal pred (-)")
axes[2].set_title(f"removal parity (R²={r_r2:.2f})")

plt.tight_layout()
plt.savefig(parity_png, dpi=220)
plt.show()

# Standalone feature-importance figure (in case you want it separately)
plt.figure(figsize=(8,4.5))
plt.bar(range(len(feature_cols)), imp[order])
plt.xticks(range(len(feature_cols)), np.array(feature_cols)[order], rotation=40, ha="right")
plt.ylabel("Importance")
plt.title("Feature Importance (removal_frac)")
plt.tight_layout()
plt.savefig(fi_png, dpi=220)
plt.show()

print(f"[OK] Saved figures:\n - {parity_png}\n - {fi_png}")

