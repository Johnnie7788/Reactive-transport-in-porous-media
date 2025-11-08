#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# -------------------------------------------------------------
# PROJECT:
# Two-Phase Flow of H2–CH4 Gas and Brine in Porous Media

# -------------------------------------------------------------

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# -------------------------------------------------------------
# 0) Project directory (your Windows path)
# -------------------------------------------------------------
project_dir = Path(r"")
project_dir.mkdir(parents=True, exist_ok=True)

# File paths
input_path     = project_dir / "h2ch4_brine_input_2d.csv"
profiles_path  = project_dir / "h2ch4_saturation_stack_2d.npz"     # compact time stack
final_csv_path = project_dir / "h2ch4_final_saturation_2d.csv"
plot_path      = project_dir / "h2ch4_saturation_2d_final.png"

# -------------------------------------------------------------
rng = np.random.default_rng(20251108)

Lx, Ly = 600.0, 200.0     # domain size [m]
Nx, Ny = 240, 80          # grid
x = np.linspace(0.0, Lx, Nx)
y = np.linspace(0.0, Ly, Ny)
dx, dy = x[1]-x[0], y[1]-y[0]

X, Y = np.meshgrid(x, y)  # [Ny, Nx]

# Porosity: layered + gentle lateral variation + noise (0.14–0.28)
phi = 0.20 \
      + 0.04*np.sin(2*np.pi*Y/60.0) \
      + 0.02*np.sin(2*np.pi*X/200.0) \
      + 0.005*rng.normal(size=(Ny, Nx))
phi = np.clip(phi, 0.14, 0.28)

# Permeability (mD): lognormal, correlated with porosity + channel streaks
logk_md = (np.log(150.0)
           + 1.0*(phi - phi.mean())/phi.std()
           + 0.4*rng.normal(size=(Ny, Nx)))
# Add “high-perm” streaks (channels)
for row in [int(Ny*0.25), int(Ny*0.55), int(Ny*0.8)]:
    logk_md[row:row+2, :] += 1.5
k_md = np.clip(np.exp(logk_md), 0.05, 3000.0)        # mD
k_m2 = k_md * 9.869233e-16                            # m^2

# Gas composition: more H2 near inlet (left), tapering right
x_H2 = np.clip(0.75 - 0.45*(X/Lx) + 0.03*rng.normal(size=(Ny, Nx)), 0.05, 0.95)

# Residual saturations
Swr, Sgr = 0.18, 0.05

# Save tidy input CSV (long-form)
flat = pd.DataFrame({
    "i": np.repeat(np.arange(Ny), Nx),
    "j": np.tile(np.arange(Nx), Ny),
    "x_m": X.ravel(),
    "y_m": Y.ravel(),
    "porosity": phi.ravel(),
    "permeability_m2": k_m2.ravel(),
    "x_H2": x_H2.ravel(),
    "Swr": Swr,
    "Sgr": Sgr
})
flat.to_csv(input_path, index=False)
print(f"[OK] Input written: {input_path}")

# -------------------------------------------------------------
# 2) Read input (to show end-to-end reproducibility)
# -------------------------------------------------------------
inp = pd.read_csv(input_path)
phi    = inp["porosity"].to_numpy().reshape(Ny, Nx)
k_m2   = inp["permeability_m2"].to_numpy().reshape(Ny, Nx)
x_H2   = inp["x_H2"].to_numpy().reshape(Ny, Nx)
Swr    = float(inp["Swr"].iloc[0])
Sgr    = float(inp["Sgr"].iloc[0])

# -------------------------------------------------------------
# 3) Two-phase model (gas + brine)
# -------------------------------------------------------------
# Viscosities (Pa·s)
mu_w  = 1.0e-3
mu_H2 = 8.9e-6
mu_CH4= 1.1e-5
mu_g  = x_H2*mu_H2 + (1.0 - x_H2)*mu_CH4   # composition-dependent gas viscosity

# Corey exponents
nw, ng = 2.0, 2.0

def Swn(Sw):
    return np.clip((Sw - Swr) / (1.0 - Swr - Sgr), 0.0, 1.0)

def krw(Sw):
    return Swn(Sw)**nw

def krg(Sw):
    return (1.0 - Swn(Sw))**ng

# Pressure gradient (diagonal flow) – keeps it 2D
dPdx, dPdy = -1.2e5, -2.0e4   # Pa/m

# Interface harmonic-average permeability
def harm(a, b):
    return 2*a*b / (a + b + 1e-30)

kx_face = np.empty((Ny, Nx+1))
ky_face = np.empty((Ny+1, Nx))
kx_face[:, 1:-1] = harm(k_m2[:, :-1], k_m2[:, 1:])
kx_face[:, 0]    = k_m2[:, 0]
kx_face[:, -1]   = k_m2[:, -1]
ky_face[1:-1, :] = harm(k_m2[:-1, :], k_m2[1:, :])
ky_face[0, :]    = k_m2[0, :]
ky_face[-1, :]   = k_m2[-1, :]

# Time controls (CFL)
lam_max = (1.0/mu_w) + (1.0/np.min(mu_g))
u_char  = max(kx_face.max()*abs(dPdx), ky_face.max()*abs(dPdy)) * lam_max
CFL     = 0.35
dt      = CFL / ((u_char/dx) + (u_char/dy) + 1e-30)

t_save  = [0.0, 6*3600, 12*3600, 1*24*3600, 3*24*3600]  # 0h, 6h, 12h, 1d, 3d
t_final = t_save[-1]
n_steps = int(np.ceil(t_final/dt))
dt      = t_final / n_steps

# Initial & BCs
Sw      = Swr*np.ones((Ny, Nx))     # initially gas-filled at residual water
Sw_left = 1.0 - Sgr                 # inject brine on the left boundary

snapshots = {}
t = 0.0

for step in range(n_steps):
    # Extend Sw for face values (ghosts for BCs)
    # x-faces (Ny, Nx+1): left ghost from injection; right ghost zero-gradient
    SwL = np.empty((Ny, Nx+1))
    SwL[:, 0]    = Sw_left
    SwL[:, 1:]   = Sw
    SwR = np.empty((Ny, Nx+1))
    SwR[:, :-1]  = Sw
    SwR[:, -1]   = Sw[:, -1]

    # y-faces (Ny+1, Nx): top/bottom zero-gradient
    SwB = np.empty((Ny+1, Nx))
    SwB[0, :]    = Sw[0, :]
    SwB[1:, :]   = Sw
    SwT = np.empty((Ny+1, Nx))
    SwT[:-1, :]  = Sw
    SwT[-1, :]   = Sw[-1, :]

    # Face viscosities (use "left/bottom" states for upwind proxy)
    mu_g_xface = np.empty((Ny, Nx+1))
    mu_g_xface[:, :-1] = mu_g
    mu_g_xface[:, -1]  = mu_g[:, -1]

    mu_g_yface = np.empty((Ny+1, Nx))
    mu_g_yface[:-1, :] = mu_g
    mu_g_yface[-1, :]  = mu_g[-1, :]

    # Total mobility at faces using left/bottom Sw
    lam_x = krw(SwL)/mu_w + krg(SwL)/mu_g_xface
    lam_y = krw(SwB)/mu_w + krg(SwB)/mu_g_yface

    # Face velocities via Darcy
    ux = -kx_face * lam_x * dPdx         # (Ny, Nx+1)
    uy = -ky_face * lam_y * dPdy         # (Ny+1, Nx)

    # Upwinded face Sw
    Sw_xface = np.where(ux >= 0.0, SwL, SwR)     # (Ny, Nx+1)
    Sw_yface = np.where(uy >= 0.0, SwB, SwT)     # (Ny+1, Nx)

    # Fractional flow of water at faces
    lamw_x = krw(Sw_xface)/mu_w
    lamg_x = krg(Sw_xface)/mu_g_xface
    fw_x   = lamw_x / (lamw_x + lamg_x + 1e-30)

    lamw_y = krw(Sw_yface)/mu_w
    lamg_y = krg(Sw_yface)/mu_g_yface
    fw_y   = lamw_y / (lamw_y + lamg_y + 1e-30)

    Fwx = ux * fw_x                             # (Ny, Nx+1)
    Fwy = uy * fw_y                             # (Ny+1, Nx)

    # Conservative update: phi*dS/dt + dFx/dx + dFy/dy = 0
    divF = (Fwx[:, 1:] - Fwx[:, :-1]) / dx + (Fwy[1:, :] - Fwy[:-1, :]) / dy
    Sw_new = Sw - (dt / phi) * divF
    Sw_new = np.clip(Sw_new, Swr, 1.0 - Sgr)

    Sw = Sw_new
    t += dt

    # Save snapshots
    for ts in t_save:
        if ts not in snapshots and abs(t - ts) <= 0.5*dt:
            snapshots[ts] = Sw.copy()

# Ensure all saves exist
for ts in t_save:
    snapshots.setdefault(ts, Sw.copy())

# -------------------------------------------------------------
# 4) Outputs (time stack + final CSV + 2D plot with legend)
# -------------------------------------------------------------
# Save compact NPZ with time snapshots
np.savez_compressed(profiles_path, x=x, y=y, times=np.array(t_save), **{f"Sw_t{int(ts)}s": snapshots[ts] for ts in t_save})

# Save final saturation in tidy CSV
final_df = pd.DataFrame({
    "i": np.repeat(np.arange(Ny), Nx),
    "j": np.tile(np.arange(Nx), Ny),
    "x_m": X.ravel(),
    "y_m": Y.ravel(),
    "Sw_final": Sw.ravel()
})
final_df.to_csv(final_csv_path, index=False)

# Professional 2D plot (heatmap + colorbar legend)
fig, ax = plt.subplots(figsize=(9, 4.6))
im = ax.imshow(Sw, origin="lower",
               extent=[x.min(), x.max(), y.min(), y.max()],
               aspect="auto")
cbar = fig.colorbar(im, ax=ax)
cbar.set_label("Brine saturation $S_w$ (–)")
ax.set_xlabel("x (m)")
ax.set_ylabel("y (m)")
ax.set_title("Two-Phase Flow (H$_2$–CH$_4$ Gas vs Brine): Final Saturation")
plt.tight_layout()
plt.savefig(plot_path, dpi=220)
plt.show()

print(f"[OK] Time stack saved: {profiles_path}")
print(f"[OK] Final saturation CSV: {final_csv_path}")
print(f"[OK] Plot saved: {plot_path}")

