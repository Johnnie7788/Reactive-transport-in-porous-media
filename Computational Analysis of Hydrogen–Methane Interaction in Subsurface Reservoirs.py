#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Project: Computational Analysis of Hydrogen–Methane Interaction in Subsurface Reservoirs


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# -----------------------------------------------------------------------------
# Settings
# -----------------------------------------------------------------------------
# Output folder. Use Path(".") to write into the current directory
# Example for Windows folder:
# out_dir = Path(r"C:\Users\YourName\Documents\PROJECT")
out_dir = Path(".")
out_dir.mkdir(parents=True, exist_ok=True)

CSV_PATH   = out_dir / "h2_ch4_pr_mixture_properties.csv"
DENS_PNG   = out_dir / "density_map.png"
ENERG_PNG  = out_dir / "energy_density_map.png"

# Reservoir temperature (K)
T = 333.15   # ~60 C

# Pressure and composition grids
P_bar_grid = np.arange(50.0, 250.0 + 1e-9, 10.0)  # bar
xH2_grid   = np.arange(0.0, 1.0 + 1e-9, 0.05)     # mol fraction

# -----------------------------------------------------------------------------
# Constants and pure-component data
# -----------------------------------------------------------------------------
R = 8.314462618  # J/mol/K

components = {
    "H2":  {"Tc": 33.19,  "Pc_bar": 12.98, "omega": -0.216, "M": 2.01588e-3},  # kg/mol
    "CH4": {"Tc": 190.56, "Pc_bar": 45.99, "omega":  0.011, "M": 16.043e-3},
}
for comp in components.values():
    comp["Pc"] = comp["Pc_bar"] * 1e5  # bar -> Pa

# Simple binary interaction parameter
kij = {("H2", "CH4"): 0.05, ("CH4", "H2"): 0.05}

# Lower heating values (MJ/kg) for volumetric energy density
LHV = {"H2": 120.0, "CH4": 50.0}

# -----------------------------------------------------------------------------
# Peng–Robinson helpers
# -----------------------------------------------------------------------------
def kappa(omega: float) -> float:
    return 0.37464 + 1.54226*omega - 0.26992*omega**2

def alpha(T: float, Tc: float, omega: float) -> float:
    tr = T / Tc
    return (1.0 + kappa(omega) * (1.0 - np.sqrt(tr)))**2

def a_i(T: float, Tc: float, Pc: float, omega: float) -> float:
    # a_i(T) for component i
    return 0.45724 * (R**2) * (Tc**2) / Pc * alpha(T, Tc, omega)

def b_i(Tc: float, Pc: float) -> float:
    return 0.07780 * R * Tc / Pc

def mix_a_b(x, a_list, b_list):
    # Quadratic mixing rule for a, linear for b
    a_mix = 0.0
    for i in range(len(x)):
        for j in range(len(x)):
            aij = np.sqrt(a_list[i] * a_list[j])
            if i != j:
                # apply symmetric binary interaction parameter
                if i == 0 and j == 1:
                    aij *= (1.0 - kij.get(("H2", "CH4"), 0.0))
                elif i == 1 and j == 0:
                    aij *= (1.0 - kij.get(("CH4", "H2"), 0.0))
            a_mix += x[i] * x[j] * aij
    b_mix = float(np.dot(x, b_list))
    return a_mix, b_mix

def cubic_pr_Z(A: float, B: float) -> float:
    # Z^3 - (1 - B) Z^2 + (A - 3B^2 - 2B) Z - (A*B - B^2 - B^3) = 0
    c2 = -(1.0 - B)
    c1 = (A - 3.0*B*B - 2.0*B)
    c0 = -(A*B - B*B - B**3)
    roots = np.roots([1.0, c2, c1, c0])
    roots_real = np.real(roots[np.isreal(roots)])
    if roots_real.size == 0:
        return np.nan
    # vapor root is the largest real root
    return float(np.max(roots_real))

# Precompute pure a and b
a_pure = {n: a_i(T, d["Tc"], d["Pc"], d["omega"]) for n, d in components.items()}
b_pure = {n: b_i(d["Tc"], d["Pc"]) for n, d in components.items()}

# -----------------------------------------------------------------------------
# Property sweep
# -----------------------------------------------------------------------------
records = []
for P_bar in P_bar_grid:
    P = P_bar * 1e5  # Pa
    for xH2 in xH2_grid:
        x = np.array([xH2, 1.0 - xH2])  # [H2, CH4]

        a_list = [a_pure["H2"], a_pure["CH4"]]
        b_list = [b_pure["H2"], b_pure["CH4"]]
        a_mix, b_mix = mix_a_b(x, a_list, b_list)

        A = a_mix * P / (R**2 * T**2)
        B = b_mix * P / (R * T)
        Z = cubic_pr_Z(A, B)

        if not np.isfinite(Z) or Z <= 0.0:
            rho = np.nan
            E_mj_m3 = np.nan
        else:
            Vm = Z * R * T / P       # m^3/mol
            M_mix = x[0]*components["H2"]["M"] + x[1]*components["CH4"]["M"]  # kg/mol
            rho = M_mix / Vm         # kg/m^3

            # mass fractions for LHV blend
            wH2  = x[0]*components["H2"]["M"]  / M_mix
            wCH4 = x[1]*components["CH4"]["M"] / M_mix

            E_mj_kg = wH2*LHV["H2"] + wCH4*LHV["CH4"]
            E_mj_m3 = rho * E_mj_kg

        records.append({
            "T_K": T,
            "P_bar": P_bar,
            "x_H2": xH2,
            "Z": Z,
            "rho_kg_per_m3": rho,
            "energy_density_MJ_per_m3": E_mj_m3
        })

df = pd.DataFrame.from_records(records)
df.to_csv(CSV_PATH, index=False)

# -----------------------------------------------------------------------------
# Heatmaps (P vs x_H2)
# -----------------------------------------------------------------------------
pivot_rho = df.pivot(index="P_bar", columns="x_H2", values="rho_kg_per_m3")
pivot_E   = df.pivot(index="P_bar", columns="x_H2", values="energy_density_MJ_per_m3")

# Density heatmap
plt.figure(figsize=(8, 5))
plt.imshow(pivot_rho.values, origin="lower",
           extent=[xH2_grid.min(), xH2_grid.max(), P_bar_grid.min(), P_bar_grid.max()],
           aspect="auto")
cbar = plt.colorbar()
cbar.set_label("Density (kg/m³)")
plt.xlabel("Hydrogen mole fraction $x_{H2}$ (–)")
plt.ylabel("Pressure (bar)")
plt.title("H$_2$–CH$_4$ Mixture Density (Peng–Robinson)")
plt.tight_layout()
plt.savefig(DENS_PNG, dpi=220)
plt.show()

# Energy density heatmap
plt.figure(figsize=(8, 5))
plt.imshow(pivot_E.values, origin="lower",
           extent=[xH2_grid.min(), xH2_grid.max(), P_bar_grid.min(), P_bar_grid.max()],
           aspect="auto")
cbar = plt.colorbar()
cbar.set_label("Energy density (MJ/m³)")
plt.xlabel("Hydrogen mole fraction $x_{H2}$ (–)")
plt.ylabel("Pressure (bar)")
plt.title("H$_2$–CH$_4$ Mixture Energy Density (Peng–Robinson)")
plt.tight_layout()
plt.savefig(ENERG_PNG, dpi=220)
plt.show()

print(f"Saved CSV: {CSV_PATH}")
print(f"Saved plot: {DENS_PNG}")
print(f"Saved plot: {ENERG_PNG}")

