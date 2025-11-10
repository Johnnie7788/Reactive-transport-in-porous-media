#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# NanoFlowNet-2D: Numerical Simulation of Fluid Flow in Nanoporous Media using the Lattice Boltzmann Method
from __future__ import annotations
import csv, sys
from pathlib import Path
from typing import Dict, Any
import numpy as np
import numpy as np
import matplotlib.pyplot as plt

CONFIG_PATH = Path(r"")
OUT_DIR     = Path(r"")

DEFAULTS = {
    "nx": 180, "ny": 120,
    "porosity": 0.55, "smoothing": 2, "seed": 42,
    "tau": 1.0,          # higher viscosity (nu=(tau-0.5)/3) => more stable
    "fx": 5e-7, "fy": 0.0,
    "rho0": 1.0,
    "steps": 2000,
    "steady_tol": 1e-7,
    "steady_window": 200,
    "progress_every": 100,
    "u_clip": 0.1        # velocity clamp for stability
}

def read_config_csv(path: Path) -> Dict[str, Any]:
    cfg = DEFAULTS.copy()
    if not path.exists():
        print(f"[info] Config not found at {path}. Using built-in defaults.")
        return cfg
    try:
        with open(path, "r", newline="") as f:
            r = csv.reader(f)
            for row in r:
                if not row or len(row) < 2: continue
                key = row[0].strip(); val = ",".join(row[1:]).strip()
                if not key or key.lower() == "key" or key.startswith("#"): continue
                if key in ("nx","ny","seed","steps","steady_window","progress_every","smoothing"):
                    try: cfg[key] = int(float(val))
                    except: pass
                elif key in ("porosity","tau","fx","fy","rho0","steady_tol","u_clip"):
                    try: cfg[key] = float(val)
                    except: pass
    except Exception as e:
        print(f"[warn] Failed reading config: {e}. Using defaults.")
    return cfg

def box_blur_2d(arr: np.ndarray, passes: int = 2) -> np.ndarray:
    arr = arr.astype(np.float64, copy=False)
    for _ in range(max(0, int(passes))):
        s = (np.roll(arr,  0, 0) + np.roll(arr,  1, 0) + np.roll(arr, -1, 0))
        s = (s + np.roll(s,  1, 1) + np.roll(s, -1, 1)) / 3.0
        arr = s / 3.0
    return arr

def generate_porosity_2d(nx, ny, porosity=0.55, smoothing=2, seed=42):
    if seed is not None:
        np.random.seed(int(seed))
    noise = np.random.rand(ny, nx)
    sm = box_blur_2d(noise, passes=int(smoothing))
    thr = np.quantile(sm, 1 - float(porosity))
    pores = (sm > thr).astype(np.uint8)
    return pores

# D2Q9
C = np.array([[0,0],[1,0],[0,1],[-1,0],[0,-1],[1,1],[-1,1],[-1,-1],[1,-1]], dtype=np.int8)
W = np.array([4/9, 1/9,1/9,1/9,1/9, 1/36,1/36,1/36,1/36], dtype=np.float64)
OPP = np.array([0,3,4,1,2,7,8,5,6], dtype=np.int32)
CS2 = 1.0/3.0
Q = 9

def initialize_fields(pores, rho0=1.0):
    ny, nx = pores.shape
    rho = np.full((ny, nx), float(rho0), dtype=np.float64)
    f = np.empty((Q, ny, nx), dtype=np.float64)
    for i in range(Q):
        f[i] = W[i] * rho
    solid = (pores == 0)
    rho[solid] = 0.0
    for i in range(Q):
        f[i][solid] = 0.0
    return rho, f

def equilibrium(rho, ux, uy):
    ny, nx = rho.shape
    feq = np.empty((Q, ny, nx), dtype=np.float64)
    u2 = ux*ux + uy*uy
    for i in range(Q):
        cx, cy = C[i]
        cu = cx*ux + cy*uy
        feq[i] = W[i] * rho * (1.0 + cu/CS2 + 0.5*(cu*cu)/(CS2*CS2) - 0.5*u2/CS2)
    return feq

def lbm_step(f, pores, tau, Fx, Fy, u_clip):
    rho = np.sum(f, axis=0)
    ux = np.zeros_like(rho); uy = np.zeros_like(rho)
    for i in range(Q):
        ux += C[i,0]*f[i]; uy += C[i,1]*f[i]
    mask = rho > 0
    ux[mask] /= rho[mask]; uy[mask] /= rho[mask]

    # forcing
    ux += 0.5*Fx; uy += 0.5*Fy

    # stability clamp on velocities (prevents overflow in feq)
    if u_clip is not None and u_clip > 0:
        ux = np.clip(ux, -u_clip, u_clip)
        uy = np.clip(uy, -u_clip, u_clip)

    omega = 1.0/float(tau)
    feq = equilibrium(rho, ux, uy)
    for i in range(Q):
        f[i] = (1.0 - omega)*f[i] + omega*feq[i] + W[i]*(3.0*(C[i,0]*Fx + C[i,1]*Fy))

    solid = (pores == 0)
    for i in range(Q):
        tmp = f[i][solid].copy()
        f[i][solid] = f[OPP[i]][solid]
        f[OPP[i]][solid] = tmp

    f_streamed = np.zeros_like(f)
    for i in range(Q):
        cx, cy = C[i]
        f_streamed[i] = np.roll(np.roll(f[i], cx, axis=1), cy, axis=0)

    return rho, ux, uy, f_streamed

def save_pgm(path: Path, img: np.ndarray):
    arr = np.nan_to_num(np.asarray(img, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    vmin = float(arr.min()); vmax = float(arr.max())
    if vmax <= vmin: vmax = vmin + 1.0
    scaled = np.clip(255.0*(arr - vmin)/(vmax - vmin), 0, 255).astype(np.uint8)
    ny, nx = scaled.shape
    with open(path, "wb") as f:
        f.write(f"P5\n{nx} {ny}\n255\n".encode("ascii"))
        f.write(scaled.tobytes())

def main():
    print("=== NanoFlowNet 2D (stable) ===")
    print(f"Python: {sys.version.split()[0]}")
    print(f"Config: {CONFIG_PATH}")

    cfg = read_config_csv(CONFIG_PATH)
    tau = float(cfg["tau"]);    tau = 0.8 if tau <= 0.5 else tau
    por = float(cfg["porosity"]);  por = 0.55 if not (0.0 < por < 1.0) else por
    u_clip = float(cfg.get("u_clip", DEFAULTS["u_clip"]))

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    nx, ny = int(cfg["nx"]), int(cfg["ny"])
    print(f"[info] Grid: {nx} x {ny}")

    pores = generate_porosity_2d(nx, ny, porosity=por,
                                 smoothing=int(cfg["smoothing"]),
                                 seed=int(cfg["seed"]))
    np.save(OUT_DIR / "porous_mask.npy", pores)
    print(f"[ok] Porous mask ready: porosity={pores.mean():.3f}")

    rho, f = initialize_fields(pores, rho0=float(cfg["rho0"]))
    Fx, Fy = float(cfg["fx"]), float(cfg["fy"])
    steps = int(cfg["steps"])
    steady_tol = float(cfg["steady_tol"])
    steady_window = max(1, int(cfg["steady_window"]))
    progress_every = max(1, int(cfg["progress_every"]))

    last_mean = None
    for step in range(1, steps + 1):
        rho, ux, uy, f = lbm_step(f, pores, tau, Fx, Fy, u_clip)

        if step % progress_every == 0:
            pm = pores == 1
            print(f"[progress] step {step:5d} | <u_x>={float(ux[pm].mean()):.3e}")

        if step % steady_window == 0:
            pm = pores == 1
            ux_mean = float(ux[pm].mean()) if pm.any() else 0.0
            if last_mean is not None and abs(ux_mean - last_mean) < steady_tol:
                print(f"[ok] Converged at step {step} (Δ={abs(ux_mean-last_mean):.2e} < {steady_tol})")
                break
            last_mean = ux_mean

        if not np.isfinite(rho).all() or not np.isfinite(ux).all() or not np.isfinite(uy).all():
            print("[warn] Non-finite values detected; stopping early.")
            break

    pm = pores == 1
    speed = np.sqrt(ux*ux + uy*uy)
    avg_ux = float(ux[pm].mean()) if pm.any() else 0.0
    nu = (tau - 0.5)/3.0
    mu = nu * float(cfg["rho0"])
    k_app = (mu * avg_ux) / (float(cfg["rho0"]) * Fx) if Fx != 0.0 else float("nan")

    np.save(OUT_DIR / "ux.npy", np.nan_to_num(ux))
    np.save(OUT_DIR / "uy.npy", np.nan_to_num(uy))
    np.save(OUT_DIR / "rho.npy", np.nan_to_num(rho))
    np.save(OUT_DIR / "speed.npy", np.nan_to_num(speed))
    save_pgm(OUT_DIR / "speed.pgm", speed)
    save_pgm(OUT_DIR / "pores.pgm", pores*255.0)

    print("\n=== Summary (lattice units) ===")
    print(f"Porosity       : {pores.mean():.6f}")
    print(f"Viscosity (nu) : {nu:.6e}")
    print(f"Avg u_x        : {avg_ux:.6e}")
    print(f"k_app (Darcy)  : {k_app:.6e}")
    print(f"Final step     : {step}")
    print(f"[ok] Saved: ux.npy, uy.npy, rho.npy, speed.npy, speed.pgm, pores.pgm in {OUT_DIR}")

if __name__ == "__main__":
    main()


# Load saved arrays
speed = np.load(r"C:\Users\15107\Downloads\NanoFlowNet\nf2d_outputs\speed.npy")
pores = np.load(r"C:\Users\15107\Downloads\NanoFlowNet\nf2d_outputs\porous_mask.npy")

plt.figure(figsize=(8, 4))
plt.subplot(1, 2, 1)
plt.title("Porous structure")
plt.imshow(pores, cmap="gray", origin="lower")

plt.subplot(1, 2, 2)
plt.title("Flow speed magnitude")
plt.imshow(speed, cmap="viridis", origin="lower")
plt.colorbar(label="Speed")

plt.tight_layout()
plt.show()

