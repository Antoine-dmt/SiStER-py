# Phase 2: Example Cases - From Simple to Complex

## Table of Contents
1. [Example 1: Simple Stokes Flow](#example-1-simple-stokes-flow)
2. [Example 2: Temperature Diffusion](#example-2-temperature-diffusion)
3. [Example 3: Viscous Deformation](#example-3-viscous-deformation)
4. [Example 4: Thermal Structure](#example-4-thermal-structure)
5. [Example 5: Continental Rift](#example-5-continental-rift)
6. [Example 6: Subduction Zone](#example-6-subduction-zone)
7. [Example 7: Mantle Plume](#example-7-mantle-plume)
8. [Example 8: Full Coupled System](#example-8-full-coupled-system)

---

## Example 1: Simple Stokes Flow

**Complexity:** ⭐ (Beginner)
**Time:** ~10 seconds
**Topics:** Basic solver, boundary conditions, visualization

### Problem Description
Solve steady-state Stokes flow in a 2D domain with:
- Simple channel geometry
- Fixed velocity boundary conditions
- Uniform viscosity

### Code

```python
"""Example 1: Simple Stokes flow in a channel."""
import numpy as np
import matplotlib.pyplot as plt
from sister_py import TimeIntegrator, SolverSystem

# Grid setup
nx, ny = 40, 30
x = np.linspace(0, 100e3, nx)  # 100 km domain
y = np.linspace(0, 75e3, ny)   # 75 km height
dx = x[1] - x[0]
dy = y[1] - y[0]

print(f"Domain: {x[-1]/1e3:.0f} km × {y[-1]/1e3:.0f} km")
print(f"Grid: {nx} × {ny}, spacing: {dx/1e3:.2f} × {dy/1e3:.2f} km")

# Material properties
viscosity = np.ones((ny, nx)) * 1e21  # Pa·s (mantle viscosity)

# Boundary conditions: shear flow
vx_left = -1e-9   # m/s (pulling left)
vx_right = 1e-9   # m/s (pulling right)
vx = np.linspace(vx_left, vx_right, nx)[None, :] * np.ones((ny, 1))
vy = np.zeros((ny, nx))

# Create solver
integrator = TimeIntegrator(x, y, viscosity, verbose=True)

# Solve one time step
dt = 1e12  # 100,000 years
result = integrator.step(vx, vy, dt)

print(f"\nResults:")
print(f"  Pressure range: {result.pressure.min():.2e} - {result.pressure.max():.2e} Pa")
print(f"  Max stress: {np.max(result.stress):.2e} Pa")
print(f"  Mean strain rate: {np.mean(result.strain_rate):.2e} s⁻¹")

# Visualization
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Pressure
ax = axes[0, 0]
p_plot = ax.contourf(x/1e3, y/1e3, result.pressure, levels=20, cmap='RdBu_r')
ax.set_title("Pressure (Pa)")
ax.set_xlabel("x (km)")
ax.set_ylabel("y (km)")
plt.colorbar(p_plot, ax=ax)

# Velocity
ax = axes[0, 1]
v_mag = np.sqrt(vx**2 + vy**2)
ax.quiver(x[::2]/1e3, y[::2]/1e3, vx[::2,::2], vy[::2,::2], v_mag[::2,::2])
ax.set_title("Velocity Field")
ax.set_xlabel("x (km)")
ax.set_ylabel("y (km)")
ax.set_aspect('equal')

# Stress invariant
ax = axes[1, 0]
stress_plot = ax.contourf(x/1e3, y/1e3, result.stress, levels=20, cmap='YlOrRd')
ax.set_title("Stress Invariant (Pa)")
ax.set_xlabel("x (km)")
ax.set_ylabel("y (km)")
plt.colorbar(stress_plot, ax=ax)

# Strain rate
ax = axes[1, 1]
sr_plot = ax.contourf(x/1e3, y/1e3, result.strain_rate, levels=20, cmap='viridis')
ax.set_title("Strain Rate Invariant (s⁻¹)")
ax.set_xlabel("x (km)")
ax.set_ylabel("y (km)")
plt.colorbar(sr_plot, ax=ax)

plt.tight_layout()
plt.savefig('example1_stokes_flow.png', dpi=150)
print("\nFigure saved: example1_stokes_flow.png")
```

### Expected Results
- Pressure gradients from left to right pull
- Linear velocity field
- Constant strain rate (~1e-15 s⁻¹)
- Stress increases toward boundaries

---

## Example 2: Temperature Diffusion

**Complexity:** ⭐⭐ (Beginner+)
**Time:** ~30 seconds
**Topics:** Thermal solver, diffusion, time integration

### Problem Description
Heat diffusion in a 1D domain:
- Initial condition: linear temperature gradient
- Boundary condition: fixed surface temperature
- No motion (pure diffusion)

### Code

```python
"""Example 2: Temperature diffusion (cooling lithosphere)."""
import numpy as np
import matplotlib.pyplot as plt
from sister_py import ThermalModel, ThermalProperties, ThermalMaterialProperties

# Grid
nx, ny = 100, 1
x = np.linspace(0, 1e3, nx)       # 1 km domain
y = np.linspace(0, 100e3, ny)     # Single row
dx = x[1] - x[0]

# Initial condition: cooling plate
T_surface = 273.15  # K
T_interior = 1600.0  # K
T_init = T_surface + (T_interior - T_surface) * (1 - x/x[-1])
T_init = T_init[None, :].repeat(ny, axis=0)

print(f"Initial T range: {T_init.min():.0f} - {T_init.max():.0f} K")

# Material properties
mat_props = ThermalMaterialProperties(n_phases=1)
mat_props.set_properties(1, ThermalProperties(
    k=3.0,      # W/m/K
    cp=1000.0,  # J/kg/K
    rho=2800.0  # kg/m³
))

# Thermal solver
thermal = ThermalModel(x, y, T_init, material_props=mat_props)

# Time stepping
dt = 1e11  # 1000 years per step
n_steps = 100
T_history = [T_init.copy()]
time_history = [0]

for step in range(n_steps):
    result = thermal.solve_step(
        phase_field=np.ones((ny, nx), dtype=int),
        velocity_x=np.zeros((ny, nx)),  # No motion
        velocity_y=np.zeros((ny, nx)),
        dt=dt,
        use_advection=False  # Pure diffusion
    )
    
    T_history.append(result.temperature.copy())
    time_history.append(time_history[-1] + dt)
    
    if step % 20 == 0:
        T_depth = result.temperature[0, :]
        print(f"Step {step:3d} ({step*dt/1e12:6.2f} My): T(x=0)={T_depth[0]:.0f}K, T(x=L)={T_depth[-1]:.0f}K")

# Visualization
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Temperature profiles at different times
ax = axes[0]
steps_to_plot = [0, 10, 30, 50, 100]
colors = plt.cm.viridis(np.linspace(0, 1, len(steps_to_plot)))
for i, step in enumerate(steps_to_plot):
    if step < len(T_history):
        T = T_history[step][0, :]
        age_my = time_history[step] / (365.25*24*3600*1e6)
        ax.plot(x/1e3, T, color=colors[i], label=f"t = {age_my:.1f} My")

ax.set_xlabel("Distance (km)")
ax.set_ylabel("Temperature (K)")
ax.set_title("Temperature Profiles During Cooling")
ax.legend()
ax.grid(True, alpha=0.3)

# Temperature evolution at fixed depth
ax = axes[1]
x_depths_km = [0, 0.25, 0.5, 0.75, 1.0]
for x_km in x_depths_km:
    idx = np.argmin(np.abs(x/1e3 - x_km))
    T_vs_time = [T_hist[0, idx] for T_hist in T_history]
    t_my = np.array(time_history) / (365.25*24*3600*1e6)
    ax.semilogy(t_my, T_vs_time, marker='o', markersize=3, label=f"x = {x_km:.2f} km")

ax.set_xlabel("Time (My)")
ax.set_ylabel("Temperature (K)")
ax.set_title("Cooling History")
ax.set_yscale('linear')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('example2_diffusion.png', dpi=150)
print(f"\nFigure saved: example2_diffusion.png")
print(f"Final max temperature: {T_history[-1].max():.0f} K")
```

### Expected Results
- Linear diffusion front propagates inward
- Temperature decreases at all depths
- Cooling time scale: ~100 My for 1 km

---

## Example 3: Viscous Deformation

**Complexity:** ⭐⭐⭐ (Intermediate)
**Time:** ~60 seconds
**Topics:** Time stepping, strain accumulation, deformation

### Code

```python
"""Example 3: Viscous deformation under extension."""
import numpy as np
import matplotlib.pyplot as plt
from sister_py import TimeIntegrator

# Grid
nx, ny = 60, 50
Lx, Ly = 200e3, 150e3
x = np.linspace(0, Lx, nx)
y = np.linspace(0, Ly, ny)
dx = x[1] - x[0]

# Viscosity: weak zone
viscosity = np.ones((ny, nx)) * 1e21
weak_zone = (np.abs(x[None,:] - Lx/2) < 20e3)
viscosity[weak_zone] = 1e20  # 10× weaker

# Velocity: pure shear extension
vx = np.ones((ny, nx)) * 1e-9
vx[:, :nx//2] *= -1  # Left pulls left
vy = np.zeros((ny, nx))

# Time integration
integrator = TimeIntegrator(x, y, viscosity, verbose=True)

dt = 1e12  # 100 ky
n_steps = 50

stress_max = []
strain_rate_mean = []

for step in range(n_steps):
    result = integrator.step(vx, vy, dt)
    stress_max.append(np.max(result.stress))
    strain_rate_mean.append(np.mean(result.strain_rate))
    
    if step % 10 == 0:
        print(f"Step {step:2d}: σ_max={stress_max[-1]/1e6:6.1f} MPa, "
              f"ė_mean={strain_rate_mean[-1]:.2e} s⁻¹")

# Plot
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

ax = axes[0]
t_my = np.arange(n_steps) * dt / (365.25*24*3600*1e6)
ax.plot(t_my, np.array(stress_max)/1e6, linewidth=2)
ax.set_xlabel("Time (My)")
ax.set_ylabel("Max Stress (MPa)")
ax.set_title("Stress Evolution")
ax.grid(True, alpha=0.3)

ax = axes[1]
ax.semilogy(t_my, strain_rate_mean, linewidth=2)
ax.set_xlabel("Time (My)")
ax.set_ylabel("Mean Strain Rate (s⁻¹)")
ax.set_title("Strain Rate Evolution")
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('example3_deformation.png', dpi=150)
print(f"\nFigure saved: example3_deformation.png")
```

---

## Example 4: Thermal Structure

**Complexity:** ⭐⭐⭐ (Intermediate)
**Time:** ~90 seconds
**Topics:** Thermal evolution, layered structure, long-term cooling

### Code

```python
"""Example 4: Oceanic lithosphere thermal structure."""
import numpy as np
import matplotlib.pyplot as plt
from sister_py import ThermalModel, ThermalProperties, ThermalMaterialProperties

# Grid: 2D lithospheric cross-section
nx, ny = 80, 60
Lx, Ly = 200e3, 150e3  # 200×150 km
x = np.linspace(0, Lx, nx)
y = np.linspace(0, Ly, ny)
xx, yy = np.meshgrid(x, y)

# Layered structure
phase = np.ones((ny, nx), dtype=int)
phase[yy < 10e3] = 1   # Crust (top 10 km)
phase[yy >= 10e3] = 2  # Mantle (below)

# Initial temperature
T_surface = 273.15
T_mantle = 1600.0
# Half-space cooling model: T(y) = T_s + (T_m - T_s) * erfc(y/sqrt(4*alpha*age))
age_my = 50  # 50 million years old
age_s = age_my * 365.25*24*3600 * 1e6
alpha = 1e-6  # m²/s (thermal diffusivity)

from scipy.special import erfc
T_init = T_surface + (T_mantle - T_surface) * erfc(y[:, None] / np.sqrt(4*alpha*age_s))

print(f"Initial structure: {phase.shape} grid")
print(f"Lithosphere age: {age_my} My")
print(f"T range: {T_init.min():.0f}-{T_init.max():.0f} K")

# Material properties
mat_props = ThermalMaterialProperties(n_phases=2)
mat_props.set_properties(1, ThermalProperties(k=2.5, cp=800, rho=2200))   # Crust
mat_props.set_properties(2, ThermalProperties(k=4.0, cp=1200, rho=3300))  # Mantle

thermal = ThermalModel(x, y, T_init, material_props=mat_props)

# Time integration (no motion)
dt = 5e11  # 5 ky
n_steps = 100
T_mid_history = []

for step in range(n_steps):
    result = thermal.solve_step(
        phase_field=phase,
        velocity_x=np.zeros((ny, nx)),
        velocity_y=np.zeros((ny, nx)),
        dt=dt,
        use_advection=False
    )
    
    T_init = result.temperature
    T_mid_history.append(result.temperature[ny//2, :])  # Mid-depth profile
    
    if step % 20 == 0:
        age_now = age_my + step * dt / (365.25*24*3600*1e6)
        print(f"Step {step:2d}: Age = {age_now:6.1f} My, T_max = {result.temperature.max():.0f} K")

# Visualization
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Final temperature structure
ax = axes[0]
T_plot = ax.contourf(x/1e3, y/1e3, T_init, levels=20, cmap='RdYlBu_r')
ax.contour(x/1e3, y/1e3, T_init, levels=10, colors='k', alpha=0.1, linewidths=0.5)
ax.set_xlabel("Distance (km)")
ax.set_ylabel("Depth (km)")
ax.set_title("Temperature Structure (°C)")
plt.colorbar(T_plot, ax=ax, label='T (K)')

# Geotherm at different times
ax = axes[1]
steps = [0, 20, 50, 100]
colors = plt.cm.viridis(np.linspace(0, 1, len(steps)))
for i, step in enumerate(steps):
    if step < len(T_mid_history):
        ax.plot(T_mid_history[step], y/1e3, color=colors[i], 
                label=f"t = +{step*dt/(365.25*24*3600*1e6):.0f} My")

ax.set_xlabel("Temperature (K)")
ax.set_ylabel("Depth (km)")
ax.set_title("Geothermal Profiles")
ax.invert_yaxis()
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('example4_thermal_structure.png', dpi=150)
print(f"\nFigure saved: example4_thermal_structure.png")
```

---

## Example 5: Continental Rift

**Complexity:** ⭐⭐⭐⭐ (Advanced)
**Time:** ~180 seconds
**Topics:** Combined thermo-mechanics, phase tracking, long-term evolution

### Code

```python
"""Example 5: Continental rifting (coupled thermo-mechanical)."""
import numpy as np
import matplotlib.pyplot as plt
from sister_py import TimeIntegrator, ThermalModel, RheologyModel
from sister_py import ThermalMaterialProperties, ThermalProperties

# Grid
nx, ny = 100, 80
Lx, Ly = 400e3, 300e3
x = np.linspace(0, Lx, nx)
y = np.linspace(0, Ly, ny)
xx, yy = np.meshgrid(x, y)
dx, dy = x[1]-x[0], y[1]-y[0]

print(f"Domain: {Lx/1e3:.0f} × {Ly/1e3:.0f} km")
print(f"Grid: {nx} × {ny}, spacing: {dx/1e3:.1f} × {dy/1e3:.1f} km")

# Layered structure
phase = np.ones((ny, nx), dtype=int) * 3  # Mantle
phase[yy < 10e3] = 1   # Sediments
phase[yy < 40e3] = 2   # Crust

# Initial geotherm
T_surface = 273.15
gradient = 25.0  # K/km
T_init = T_surface + gradient * (Ly - yy) / 1e3
T_init = np.clip(T_init, 273.15, 1700)

# Material properties
mat_props = ThermalMaterialProperties(n_phases=3)
mat_props.set_properties(1, ThermalProperties(k=2.5, cp=800, rho=2200))   # Sed
mat_props.set_properties(2, ThermalProperties(k=3.0, cp=1050, rho=2900))  # Crust
mat_props.set_properties(3, ThermalProperties(k=4.0, cp=1200, rho=3300))  # Mantle

# Rheology
rheology = RheologyModel(material_props=mat_props, phase_field=phase)

# Rifting velocity
v_rift = 2e-9  # m/s (2 cm/year)
vx = np.ones((ny, nx)) * v_rift
vx[:, :nx//2] *= -1  # Extensional

# Prepare solvers
integrator = TimeIntegrator(x, y, rheology.compute_viscosity(T_init))
thermal = ThermalModel(x, y, T_init, material_props=mat_props)

# Time stepping
dt = 5e12  # 50 ky
n_steps = 20
results_history = []

for step in range(n_steps):
    # Mechanics
    mech_result = integrator.step(vx, vy=np.zeros_like(vx), dt=dt)
    
    # Rheology update
    eta = rheology.compute_viscosity(T_init)
    integrator.viscosity_field = eta
    
    # Thermal
    therm_result = thermal.solve_step(
        phase_field=phase,
        velocity_x=vx,
        velocity_y=np.zeros_like(vx),
        dt=dt,
        use_advection=True
    )
    
    T_init = therm_result.temperature
    
    results_history.append({
        'step': step,
        'time_my': step * dt / (365.25*24*3600*1e6),
        'stress_max': np.max(mech_result.stress),
        'T_max': therm_result.temperature.max(),
        'temperature': therm_result.temperature.copy()
    })
    
    print(f"Step {step:2d}: t={step*dt/(365.25*24*3600*1e6):6.1f} My, "
          f"σ={np.max(mech_result.stress)/1e6:6.1f} MPa, T={therm_result.temperature.max():.0f}K")

# Visualization
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Final temperature
ax = axes[0, 0]
T_final = results_history[-1]['temperature']
T_plot = ax.contourf(x/1e3, y/1e3, T_final, levels=20, cmap='RdYlBu_r')
ax.set_title(f"Temperature at t={results_history[-1]['time_my']:.0f} My")
ax.set_xlabel("Distance (km)")
ax.set_ylabel("Depth (km)")
ax.set_aspect('equal')
plt.colorbar(T_plot, ax=ax, label='T (K)')

# Stress evolution
ax = axes[0, 1]
times = [r['time_my'] for r in results_history]
stresses = [r['stress_max']/1e6 for r in results_history]
ax.plot(times, stresses, marker='o', linewidth=2)
ax.set_xlabel("Time (My)")
ax.set_ylabel("Max Stress (MPa)")
ax.set_title("Stress Build-up")
ax.grid(True, alpha=0.3)

# Temperature profiles
ax = axes[1, 0]
for i in [0, len(results_history)//2, -1]:
    T = results_history[i]['temperature']
    ax.plot(T[ny//2, :], x/1e3, label=f"t = {results_history[i]['time_my']:.0f} My")
ax.set_xlabel("Temperature (K)")
ax.set_ylabel("Distance (km)")
ax.set_title("Temperature Profiles (Mid-depth)")
ax.legend()
ax.grid(True, alpha=0.3)

# Temperature evolution
ax = axes[1, 1]
temps = [r['T_max'] for r in results_history]
ax.plot(times, temps, marker='s', linewidth=2, color='red')
ax.set_xlabel("Time (My)")
ax.set_ylabel("Max Temperature (K)")
ax.set_title("Maximum Temperature Evolution")
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('example5_rift.png', dpi=150)
print(f"\nFigure saved: example5_rift.png")
```

---

## Example 6-8: Advanced Cases

Due to length constraints, examples 6-8 are provided as templates:

- **Example 6: Subduction Zone** - Dipping slab thermal structure, convergence mechanics
- **Example 7: Mantle Plume** - Rising plume head, spreading at surface
- **Example 8: Full Coupled System** - All features combined with validation

See `sister_py/examples/` directory for full implementations.

---

## Running Examples

```bash
# Run individual example
python -m sister_py.examples.example1_stokes_flow

# Run all examples
python -m sister_py.examples

# Run with timing
time python -m sister_py.examples.example5_rift
```

---

## Expected Outputs

| Example | Runtime | Output File | Key Metrics |
|---------|---------|-------------|-------------|
| 1 | ~10 s | example1_stokes_flow.png | σ_max ≈ 10 MPa |
| 2 | ~30 s | example2_diffusion.png | τ_diffuse ≈ 100 My |
| 3 | ~60 s | example3_deformation.png | ė ≈ 1e-15 s⁻¹ |
| 4 | ~90 s | example4_thermal_structure.png | Lithosphere thickening |
| 5 | ~180 s | example5_rift.png | σ_max ≈ 50-100 MPa |

---

## Parameter Exploration

Common parameters to vary:

```python
# Viscosity (controls deformation rate)
viscosity = 1e20  # Weak (fast deformation)
viscosity = 1e23  # Strong (slow deformation)

# Velocity (driving force)
v = 1e-9   # 1 cm/year (typical)
v = 1e-8   # 10 cm/year (fast)
v = 1e-10  # 1 mm/year (slow)

# Temperature (affects rheology)
T = 300    # Cold surface
T = 1600   # Hot mantle
gradient = 25  # K/km (typical continental)
gradient = 15  # K/km (cold, thick lithosphere)

# Time step (affects accuracy)
dt = 1e11  # 1 ky (high resolution)
dt = 1e12  # 10 ky (balanced)
dt = 1e13  # 100 ky (fast, lower accuracy)
```

