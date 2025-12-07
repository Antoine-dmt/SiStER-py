#!/usr/bin/env python3
"""
Simple Ridge Deformation Execution Script
Executes Ridge_Mantle_Flow.ipynb step by step with error handling
"""

import json
import sys
import os
import traceback
from pathlib import Path

# Add workspace to path
sys.path.insert(0, str(Path.cwd()))

# Import needed libraries
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from scipy.integrate import odeint
import warnings

warnings.filterwarnings('ignore')

# Set up plotting
plt.rcParams['figure.dpi'] = 100
plt.rcParams['font.size'] = 9

print("=" * 80)
print("🌊 RIDGE MANTLE FLOW: BRITTLE-DUCTILE DEFORMATION SIMULATION")
print("=" * 80)

# =============================================================================
# PART 1: DOMAIN SETUP
# =============================================================================
print("\n[1/9] Setting up domain...")

# Domain dimensions
x_min, x_max = -100e3, 100e3  # m
z_min, z_max = -150e3, 0      # m (depth positive down)
nx, nz = 120, 100

# Create grid
x_grid = np.linspace(x_min, x_max, nx)
z_grid = np.linspace(z_min, z_max, nz)
xx, zz = np.meshgrid(x_grid, z_grid)

# Material parameters
rho = 2900  # kg/m^3
g = 9.81    # m/s^2
mu_friction = 0.6
C_0_ref = 10e6  # Pa (reference cohesion)
T_melt = 1600   # K
kappa = 1e-6    # m^2/s (thermal diffusivity)
v_spread = 1e-3 / (365.25 * 24 * 3600)  # m/s (1 mm/yr)
z_brittle = 12e3  # m (BDT depth)
T_mantle_adiabat = 1350  # K

print(f"  Domain: {(x_max-x_min)/1e3:.0f} × {(z_max-z_min)/1e3:.0f} km")
print(f"  Grid: {nx} × {nz} cells")
print(f"  Spreading rate: {v_spread*365.25*24*3600*1e3:.1f} mm/yr")
print(f"  BDT depth: {z_brittle/1e3:.0f} km")

# =============================================================================
# PART 2: RHEOLOGY FUNCTIONS
# =============================================================================
print("\n[2/9] Setting up rheology functions...")

def compute_viscosity_hk03(T, strain_rate_II, n=3.5, E_a=530e3, A_n=6.4e-28, R=8.314):
    """Hirth & Kohlstedt 2003 flow law for dry olivine"""
    # Avoid division by zero
    str_rate = np.maximum(strain_rate_II, 1e-20)
    
    # Compute viscosity: eta = (A_n)^(-1/n) * (strain_rate)^((1-n)/n) * exp(E_a/nRT)
    exp_term = np.exp(E_a / (n * R * T))
    eta = (A_n ** (-1/n)) * (str_rate ** ((1-n)/n)) * exp_term
    
    return np.clip(eta, 1e23, 1e26)

def yield_strength_coulomb(z, T):
    """Drucker-Prager/Coulomb yield strength"""
    cohesion = C_0_ref * np.maximum(1 - T / T_melt, 0.01)
    lithostatic = mu_friction * rho * g * np.abs(z)
    tau_yield = cohesion + lithostatic
    return tau_yield

print("  ✓ Hirth & Kohlstedt 2003 (dry olivine)")
print("  ✓ Coulomb friction yield criterion")

# =============================================================================
# PART 3: THERMAL STRUCTURE
# =============================================================================
print("\n[3/9] Computing thermal structure with cool ridge...")

T_surface = 273  # K
T_axis_surface = 400  # K (cool hydrothermal ridge)
T_init = np.zeros_like(xx)

for j in range(nz):
    for i in range(nx):
        z_loc = zz[j, i]
        x_loc = xx[j, i]
        
        # Distance from ridge axis
        dist_from_axis = np.abs(x_loc)
        
        if z_loc < z_brittle:
            # Brittle zone: linear increase from axis surface to BDT
            frac = np.abs(z_loc) / z_brittle
            T_init[j, i] = T_axis_surface + (T_mantle_adiabat - T_axis_surface) * frac
        else:
            # Ductile zone: half-space cooling + adiabatic at bottom
            age_Ma = dist_from_axis / (v_spread * 1e6 * 365.25 * 24 * 3600) / 1e6
            age_yr = age_Ma * 1e6
            
            z_below_bdt = np.abs(z_loc) - z_brittle
            
            if age_yr > 0:
                erf_arg = z_below_bdt / (2 * np.sqrt(kappa * age_yr * 365.25 * 24 * 3600))
                from scipy.special import erf
                erf_val = erf(erf_arg)
                T_init[j, i] = T_mantle_adiabat * (1 - erf_val)
            else:
                T_init[j, i] = T_mantle_adiabat

T_init = np.clip(T_init, T_surface, T_melt)

print(f"  Ridge axis surface: {T_axis_surface:.0f} K (~{T_axis_surface-273:.0f}°C)")
print(f"  BDT temperature: {T_init[int(nz*12/150), nx//2]:.0f} K")
print(f"  Max temperature: {T_init.max():.0f} K")

# =============================================================================
# PART 4: BRITTLE YIELD STRENGTH
# =============================================================================
print("\n[4/9] Computing brittle yield strength (Coulomb)...")

tau_yield_field = yield_strength_coulomb(zz, T_init)

print(f"  Min yield strength: {tau_yield_field.min()/1e6:.1f} MPa (cold surface)")
print(f"  Max yield strength: {tau_yield_field.max()/1e6:.1f} MPa (hot depth)")
print(f"  Yield strength at 12 km: {yield_strength_coulomb(z_brittle, T_mantle_adiabat)/1e6:.1f} MPa")

# =============================================================================
# PART 5: COMPOSITE RHEOLOGY
# =============================================================================
print("\n[5/9] Computing composite brittle-ductile rheology...")

# Example strain rate
strain_rate_field = 1e-14 * np.ones_like(xx)

# Ductile viscosity
eta_ductile = compute_viscosity_hk03(T_init, strain_rate_field)

# Brittle viscosity
eta_brittle = tau_yield_field / (2 * np.maximum(strain_rate_field, 1e-20))

# Composite (use minimum = yield envelope effect)
eta_composite = np.minimum(eta_ductile, eta_brittle)

print(f"  Ductile viscosity range: {eta_ductile.min():.2e} to {eta_ductile.max():.2e} Pa·s")
print(f"  Brittle viscosity range: {eta_brittle.min():.2e} to {eta_brittle.max():.2e} Pa·s")
print(f"  Composite viscosity range: {eta_composite.min():.2e} to {eta_composite.max():.2e} Pa·s")

# =============================================================================
# PART 6: TIME-DEPENDENT EVOLUTION WITH DEFORMATION TRACKING
# =============================================================================
print("\n[6/9] Running 1.5 Myr time-dependent evolution with fault tracking...")

dt = 10 * 365.25 * 24 * 3600  # 10 kyr timestep
num_steps = 150  # 1.5 Myr total
time_array = np.arange(num_steps) * dt / (1e6 * 365.25 * 24 * 3600)

# Storage for history
brittle_strain_hist = []
ductile_strain_hist = []
stress_peak_hist = []
yield_exceeded_hist = []

T_curr = T_init.copy()
u_curr = np.zeros_like(xx)
v_curr = np.zeros_like(xx)

for step in range(num_steps):
    # Velocity boundary conditions (pure shear spreading)
    left_side = x_grid < 0
    right_side = x_grid >= 0
    v_curr[:, left_side] = -v_spread
    v_curr[:, right_side] = v_spread
    
    # Compute strain rate from velocity
    dv_dx = np.gradient(v_curr, x_grid[1] - x_grid[0], axis=1)
    strain_rate_II_curr = np.abs(dv_dx)
    
    # Update rheology
    eta_ductile_curr = compute_viscosity_hk03(T_curr, strain_rate_II_curr)
    tau_yield_curr = yield_strength_coulomb(zz, T_curr)
    eta_brittle_curr = tau_yield_curr / (2 * np.maximum(strain_rate_II_curr, 1e-20))
    eta_curr = np.minimum(eta_ductile_curr, eta_brittle_curr)
    
    # Compute stress
    tau_II_curr = 2 * eta_curr * strain_rate_II_curr
    
    # Track deformation
    brittle_zone = zz < z_brittle
    yield_exceeded = tau_II_curr > tau_yield_curr * 0.9
    
    brittle_deformation = brittle_zone & yield_exceeded
    brittle_frac = np.sum(brittle_deformation) / np.maximum(np.sum(brittle_zone), 1) * 100
    brittle_strain_hist.append(brittle_frac)
    
    # Ductile deformation (where ductile rheology dominates)
    ductile_zone = zz >= z_brittle
    ductile_deformation = ductile_zone & ~(eta_curr >= eta_ductile_curr * 0.9)
    ductile_frac = np.sum(ductile_deformation) / np.maximum(np.sum(ductile_zone), 1) * 100
    ductile_strain_hist.append(ductile_frac)
    
    stress_peak_hist.append(tau_II_curr.max() / 1e6)
    
    # Simple thermal update (advection)
    T_curr += v_curr * np.gradient(T_curr, x_grid[1] - x_grid[0], axis=1) * dt
    T_curr = np.clip(T_curr, 0, T_melt)
    
    # Progress output
    if (step + 1) % 15 == 0:
        eta_to_go = (num_steps - step - 1) / 15 * 30
        print(f"  Step {step+1:3d}/{num_steps} | t={time_array[step]:6.2f} Myr | "
              f"Brittle={brittle_frac:5.1f}% | Ductile={ductile_frac:5.1f}% | "
              f"τ_max={tau_II_curr.max()/1e6:6.1f} MPa | ETA ~{eta_to_go:.0f}s")

print(f"  ✓ Simulation completed: {time_array[-1]:.2f} Myr")
print(f"  Peak brittle activity: {max(brittle_strain_hist):.1f}%")
print(f"  Peak ductile activity: {max(ductile_strain_hist):.1f}%")

# =============================================================================
# PART 7: VISUALIZATION
# =============================================================================
print("\n[7/9] Creating 6-panel visualization...")

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle('Ridge Mantle Flow: Brittle-Ductile Deformation Evolution', fontsize=16, fontweight='bold')

# Panel 1: Initial Temperature
ax = axes[0, 0]
im1 = ax.contourf(xx/1e3, zz/1e3, T_init, levels=20, cmap='RdYlBu_r')
ax.axhline(y=-z_brittle/1e3, color='white', linestyle='--', linewidth=2, label='BDT')
ax.set_title('Initial Temperature Structure', fontweight='bold')
ax.set_xlabel('Distance (km)')
ax.set_ylabel('Depth (km)')
ax.legend()
plt.colorbar(im1, ax=ax, label='Temperature (K)')

# Panel 2: Yield Strength Field
ax = axes[0, 1]
im2 = ax.contourf(xx/1e3, zz/1e3, tau_yield_field/1e6, levels=20, cmap='RdYlBu')
ax.set_title('Yield Strength (Coulomb)', fontweight='bold')
ax.set_xlabel('Distance (km)')
ax.set_ylabel('Depth (km)')
plt.colorbar(im2, ax=ax, label='Yield Strength (MPa)')

# Panel 3: Composite Viscosity Field
ax = axes[0, 2]
im3 = ax.contourf(xx/1e3, zz/1e3, np.log10(eta_composite), levels=20, cmap='viridis')
ax.set_title('Composite Rheology (brittle-ductile)', fontweight='bold')
ax.set_xlabel('Distance (km)')
ax.set_ylabel('Depth (km)')
plt.colorbar(im3, ax=ax, label='log₁₀(η) [Pa·s]')

# Panel 4: Brittle Deformation Evolution
ax = axes[1, 0]
ax.plot(time_array, brittle_strain_hist, color='darkred', linewidth=2.5, marker='o', markersize=3)
ax.fill_between(time_array, brittle_strain_hist, alpha=0.3, color='red')
ax.set_title('Brittle Deformation Activity', fontweight='bold')
ax.set_xlabel('Time (Myr)')
ax.set_ylabel('% of Brittle Zone Yielding')
ax.grid(True, alpha=0.3)
ax.set_ylim([0, max(brittle_strain_hist) * 1.1 if brittle_strain_hist else 10])

# Panel 5: Ductile Deformation Evolution
ax = axes[1, 1]
ax.plot(time_array, ductile_strain_hist, color='darkblue', linewidth=2.5, marker='s', markersize=3)
ax.fill_between(time_array, ductile_strain_hist, alpha=0.3, color='blue')
ax.set_title('Ductile Deformation Activity', fontweight='bold')
ax.set_xlabel('Time (Myr)')
ax.set_ylabel('% of Ductile Zone Deforming')
ax.grid(True, alpha=0.3)
ax.set_ylim([0, max(ductile_strain_hist) * 1.1 if ductile_strain_hist else 10])

# Panel 6: Stress Evolution
ax = axes[1, 2]
ax.plot(time_array, stress_peak_hist, color='darkgreen', linewidth=2.5, marker='^', markersize=3)
ax.fill_between(time_array, stress_peak_hist, alpha=0.3, color='green')
ax.set_title('Maximum Stress Evolution', fontweight='bold')
ax.set_xlabel('Time (Myr)')
ax.set_ylabel('Maximum Stress (MPa)')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('ridge_brittle_ductile_evolution.png', dpi=150, bbox_inches='tight')
print("  ✓ Saved: ridge_brittle_ductile_evolution.png")

# =============================================================================
# PART 8: ANALYSIS
# =============================================================================
print("\n[8/9] Analyzing deformation patterns...")

print(f"\n  BRITTLE ZONE ANALYSIS (0-12 km):")
print(f"    Peak activity: {max(brittle_strain_hist):.1f}%")
print(f"    Mean activity: {np.mean(brittle_strain_hist):.1f}%")
print(f"    Time to peak: {time_array[np.argmax(brittle_strain_hist)]:.2f} Myr")

print(f"\n  DUCTILE ZONE ANALYSIS (12+ km):")
print(f"    Peak activity: {max(ductile_strain_hist):.1f}%")
print(f"    Mean activity: {np.mean(ductile_strain_hist):.1f}%")
print(f"    Time to peak: {time_array[np.argmax(ductile_strain_hist)]:.2f} Myr")

print(f"\n  STRESS EVOLUTION:")
print(f"    Initial max stress: {stress_peak_hist[0]:.1f} MPa")
print(f"    Peak stress: {max(stress_peak_hist):.1f} MPa")
print(f"    Final stress: {stress_peak_hist[-1]:.1f} MPa")

print("\n[9/9] ✅ SIMULATION COMPLETE!\n")
print("=" * 80)
print("Results saved to:")
print("  - ridge_brittle_ductile_evolution.png (6-panel visualization)")
print("=" * 80)
