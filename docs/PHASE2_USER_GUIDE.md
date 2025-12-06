# Phase 2: User Guide - Time Stepping, Rheology & Thermal Coupling

## Table of Contents
1. Getting Started
2. Setting Up a Basic Simulation
3. Time Stepping Fundamentals
4. Rheology Configuration
5. Thermal Coupling
6. Performance Tuning
7. Common Examples
8. Troubleshooting

---

## 1. Getting Started

### Installation
```bash
# Clone and install SiSteR-py
git clone <repo>
cd SiSteR-py
pip install -e .
```

### Quick Import
```python
from sister_py import (
    TimeIntegrator, AdvectionScheme,
    RheologyModel, ThermalModel,
    OptimizedSolver,
    PoiseuilleFlow, ConvergenceStudy
)
import numpy as np
```

### Minimal Example (5 time steps)
```python
# Set up grid
nx, ny = 50, 50
x = np.linspace(0, 100e3, nx)  # 100 km domain
y = np.linspace(0, 100e3, ny)
xx, yy = np.meshgrid(x, y)

# Initialize velocity (circular flow)
vx = -2 * np.pi * yy / (100e3) * 1e-9  # m/s
vy = 2 * np.pi * xx / (100e3) * 1e-9

# Create time integrator
integrator = TimeIntegrator(
    grid_x=x, grid_y=y,
    viscosity_field=np.ones((ny, nx)) * 1e21
)

# Time stepping
dt = 1e13  # 100,000 years
for step in range(5):
    result = integrator.step(
        velocity_x=vx, velocity_y=vy, dt=dt
    )
    print(f"Step {step}: max_stress = {np.max(result.stress):.2e} Pa")
```

---

## 2. Setting Up a Basic Simulation

### Step 1: Create Grid
```python
import numpy as np

# Define domain (SI units always)
Lx = 200e3  # Width 200 km
Ly = 150e3  # Height 150 km
nx = 100    # Grid points
ny = 75

# Create grid
x = np.linspace(0, Lx, nx)
y = np.linspace(0, Ly, ny)
xx, yy = np.meshgrid(x, y)

# Grid spacing
dx = x[1] - x[0]  # ~2 km
dy = y[1] - y[0]  # ~2 km
print(f"Grid spacing: {dx/1e3:.1f} km × {dy/1e3:.1f} km")
```

### Step 2: Define Material Properties
```python
from sister_py import ThermalProperties, ThermalMaterialProperties

# Create material properties manager
material_props = ThermalMaterialProperties(n_phases=3)

# Phase 1: Sediments
material_props.set_properties(1, ThermalProperties(
    k=2.5,      # Thermal conductivity W/m/K
    cp=800.0,   # Heat capacity J/kg/K
    rho=2200.0  # Density kg/m³
))

# Phase 2: Basaltic crust
material_props.set_properties(2, ThermalProperties(
    k=3.0,
    cp=1050.0,
    rho=2900.0
))

# Phase 3: Mantle
material_props.set_properties(3, ThermalProperties(
    k=4.0,
    cp=1200.0,
    rho=3300.0
))

# Create phase field on grid
phase_field = np.ones((ny, nx), dtype=int) * 3  # Mantle everywhere
phase_field[40:, :] = 2  # Crust layer
phase_field[60:, :] = 1  # Sediment layer
```

### Step 3: Initialize Temperature
```python
# Geothermal gradient: 25 K/km (typical)
T_surface = 273.15  # K
gradient = 25.0  # K/km
T_init = T_surface + gradient * (Ly - yy) / 1e3

print(f"Temperature range: {T_init.min():.0f} - {T_init.max():.0f} K")
```

### Step 4: Set Up Solver
```python
from sister_py import RheologyModel, TimeIntegrator

# Create rheology model
rheology = RheologyModel(
    material_props=material_props,
    phase_field=phase_field
)

# Create time integrator
integrator = TimeIntegrator(
    grid_x=x, grid_y=y,
    viscosity_field=rheology.get_viscosity(T_init),
    verbose=True
)

# Thermal model
from sister_py import ThermalModel
thermal = ThermalModel(
    grid_x=x, grid_y=y,
    initial_temperature=T_init,
    material_props=material_props
)
```

### Step 5: Run Simulation
```python
# Time stepping parameters
dt = 5e12  # 50,000 years
n_steps = 100

# Storage
T_history = [T_init.copy()]
max_v_history = []

for step in range(n_steps):
    # Solve Stokes (get velocity)
    v_result = integrator.step(dt=dt)
    
    # Update rheology with temperature
    eta = rheology.compute_viscosity(T_init)
    integrator.viscosity_field = eta
    
    # Solve thermal
    T_result = thermal.solve_step(
        phase_field=phase_field,
        velocity_x=v_result.velocity_x,
        velocity_y=v_result.velocity_y,
        dt=dt,
        use_advection=True
    )
    
    # Update for next step
    T_init = T_result.temperature.copy()
    T_history.append(T_init.copy())
    max_v_history.append(np.max(np.sqrt(
        v_result.velocity_x**2 + v_result.velocity_y**2
    )))
    
    if step % 10 == 0:
        print(f"Step {step:3d}: T_max={T_init.max():.0f}K, v_max={max_v_history[-1]:.2e}m/s")

print("Simulation complete!")
```

---

## 3. Time Stepping Fundamentals

### Understanding dt (Time Step)

**What is dt?**
```python
# dt is the time step in seconds
dt_years = 1e12  # Seconds
dt_my = dt_years / (365.25*24*3600)  # Convert to millions of years
print(f"dt = {dt_my:.1f} million years")  # dt ≈ 31.7 My
```

**Choosing dt:**
```python
# For diffusion-dominated problems (thermal, pressure)
# Use larger dt (implicit method is unconditionally stable)
dt_thermal = 1e13  # 300,000 years OK

# For advection-dominated problems (marker movement)
# Use smaller dt (CFL stability limit)
max_velocity = 1e-9  # m/s (typical mantle)
dx = x[1] - x[0]    # Grid spacing
dt_advection_cfl = 0.25 * dx / max_velocity
print(f"CFL limit: dt ≤ {dt_advection_cfl/1e12:.1f} My")

# Use dt slightly less than CFL for safety
dt = min(dt_thermal, 0.9 * dt_advection_cfl)
```

### Forward Euler vs Backward Euler

```python
# Forward Euler (explicit, faster per step)
# T^{n+1} = T^n + dt * (dT/dt)
# PROS: Simple, no linear solve
# CONS: CFL limited (dt ~ dx²), unstable for large dt

# Backward Euler (implicit, robust)
# T^{n+1} = T^n + dt * (dT/dt @ T^{n+1})
# PROS: Unconditionally stable, allows large dt
# CONS: Requires solving linear system each step

# In SiSteR-py thermal solver: Always backward Euler
```

### Adaptive Time Stepping

```python
from sister_py import TimeIntegrator

integrator = TimeIntegrator(
    grid_x=x, grid_y=y,
    viscosity_field=eta,
    adaptive_timestepping=True  # Enable adaptive dt
)

# System automatically adjusts dt based on:
# - Convergence of iterative solvers
# - Changes in velocity magnitude
# - Thermal time scale
for step in range(n_steps):
    result = integrator.step(dt_initial=5e12)
    # Actual dt used may differ from initial estimate
    actual_dt = result.dt_used
    print(f"Step {step}: dt used = {actual_dt/1e12:.1f} My")
```

---

## 4. Rheology Configuration

### Temperature-Dependent Viscosity

**Arrhenius Law:**
```python
from sister_py import RheologyModel, TemperatureDependentViscosity

# Typical mantle olivine
rheology = RheologyModel()
visc_law = TemperatureDependentViscosity(
    eta_0=1e21,       # Reference viscosity at T_ref (Pa·s)
    T_ref=1273.15,    # Reference temperature (K)
    E_a=500e3,        # Activation energy (J/mol)
    R=8.314,          # Gas constant (J/mol/K)
    n=3,              # Power-law exponent (Newtonian: n=1)
    verbose=True
)

# Temperature field
T = np.linspace(300, 1600, 100)  # K
eta = visc_law.viscosity(T, strain_rate=1e-15)

import matplotlib.pyplot as plt
plt.semilogy(T, eta)
plt.xlabel("Temperature (K)")
plt.ylabel("Viscosity (Pa·s)")
plt.title("Temperature-Dependent Viscosity")
plt.show()
```

**Interpreting Results:**
```
T = 300 K (surface):    η ≈ 1e24 Pa·s (stiff, cold)
T = 1000 K:             η ≈ 1e21 Pa·s (typical mantle)
T = 1600 K (hot):       η ≈ 1e19 Pa·s (weak, hot)

Key: Cooler rocks flow ~100,000× slower than hot rocks!
```

### Pressure-Dependent Viscosity

```python
# Depth-dependent: P = ρ*g*depth
depth_km = np.array([0, 50, 100, 200, 300])
rho = 3300  # kg/m³
g = 9.81
pressure = rho * g * (depth_km * 1e3)

# Typical crustal rocks: log(η) ≈ log(η0) - 0.001 * (P - P_ref)
viscosity = np.array([1e21, 1e21.5, 1e22, 1e22.5, 1e23])

for d, p, e in zip(depth_km, pressure, viscosity):
    print(f"Depth {d:3.0f} km: P = {p/1e9:4.1f} GPa, η = {e:.1e} Pa·s")
```

### Yield Stress & Plasticity

```python
from sister_py import YieldCriterion

# Drucker-Prager yield criterion: τ_yield = C + μ*P
yield_crit = YieldCriterion(
    cohesion=10e6,      # Cohesion in Pa (typical: 1-30 MPa)
    friction_coeff=0.6  # Friction coefficient (typical: 0.3-0.8)
)

# For a stress state
pressure = 100e6  # Pa (confining pressure)
strain_rate_II = 1e-14  # s⁻¹ (second invariant)

# Get effective viscosity (with yield cutoff)
eta_eff = yield_crit.effective_viscosity(
    pressure=pressure,
    strain_rate_II=strain_rate_II,
    background_viscosity=1e21
)

print(f"Background viscosity: 1e21 Pa·s")
print(f"Effective (capped): {eta_eff:.2e} Pa·s")
```

**Physical Interpretation:**
```
Weak fault (cohesion ≈ 1 MPa):
- Yields at low stress → acts as weak plane
- Controls fault rupture and earthquakes
- Unrealistic for steady-state flow (use large C)

Strong rock (cohesion ≈ 10-30 MPa):
- Yields only at deep levels
- Controls lithosphere thickness
- Typical for geodynamics
```

### Elastic Stress Accumulation

```python
from sister_py import ElasticStressAccumulation

# Maxwell rheology: σ̇ = 2G(ė - σ/(2η))
elastic = ElasticStressAccumulation(
    shear_modulus=30e9,    # GPa (typical mantle)
    viscosity=1e21
)

# For a deformation rate
strain_rate = 1e-15  # s⁻¹
dt = 1e12           # seconds

# Update stress over time step
sigma_old = 0       # Initial stress
sigma_new = elastic.update(
    strain_rate=strain_rate,
    stress_old=sigma_old,
    dt=dt
)

# Relaxation time: τ_relax = η/G
tau_relax = elastic.relaxation_time()
print(f"Relaxation time: {tau_relax/1e13:.1f} billion years")
print(f"Mantle: Time scale >> 1 Gy, elastic effects minimal")
print(f"Lithosphere: Time scale ~ 100 My, elasticity important")
```

---

## 5. Thermal Coupling

### Setting Boundary Conditions

```python
from sister_py import ThermalBoundaryCondition, ThermalModel

thermal = ThermalModel(grid_x=x, grid_y=y, initial_temperature=T_init)

# Type 1: Dirichlet BC (fixed temperature)
bc_surface = ThermalBoundaryCondition(
    boundary='top',
    bc_type='dirichlet',
    value=273.15  # Surface at 0°C
)

# Type 2: Neumann BC (fixed heat flux)
# Typical: 60 mW/m² = 0.06 W/m²
heat_flux = 0.06  # W/m²
k_crustal = 2.5   # W/m/K
gradient = heat_flux / k_crustal
bc_bottom = ThermalBoundaryCondition(
    boundary='bottom',
    bc_type='neumann',
    value=heat_flux
)

# Type 3: Robin BC (convective)
bc_sides = ThermalBoundaryCondition(
    boundary='left',
    bc_type='robin',
    ambient_temp=273.15,
    h_coeff=50  # Convection coefficient W/m²/K
)

thermal.set_boundary_conditions([bc_surface, bc_bottom])
```

### Pure Diffusion (No Motion)

```python
# Scenario: Cooling lithospheric plate (static)
integrator_static = TimeIntegrator(grid_x=x, grid_y=y, 
                                   viscosity_field=np.ones_like(T_init)*1e23)

T = T_init.copy()
for step in range(100):
    result = thermal.solve_step(
        phase_field=phase_field,
        velocity_x=np.zeros_like(T),  # No motion
        velocity_y=np.zeros_like(T),
        heat_source=None,              # No internal heating
        dt=1e12,
        use_advection=False            # Pure diffusion
    )
    T = result.temperature
    
    if step % 20 == 0:
        print(f"Step {step}: T_max = {T.max():.0f} K")
```

**Expected Behavior:**
```
Time: 0 My    T_max ≈ 1600 K (hot interior)
Time: 50 My   T_max ≈ 1400 K (cooling)
Time: 100 My  T_max ≈ 1200 K (lithosphere thickens)
Time: 200 My  T_max ≈ 1000 K (cold, thickens further)
```

### Advection-Diffusion (Flow + Heat)

```python
# Scenario: Mantle plume (hot rising material)

# Create plume velocity (rising cylinder)
plume_center = (Lx/4, 0)  # Center at x=50km, bottom
radius = 10e3             # 10 km radius
vx_plume = np.zeros((ny, nx))
vy_plume = np.zeros((ny, nx))

for j in range(ny):
    for i in range(nx):
        r = np.sqrt((x[i] - plume_center[0])**2 + (yy[j, 0] - plume_center[1])**2)
        if r < radius:
            # Velocity magnitude decreases with distance
            v_mag = 5e-9 * (1 - r/radius)  # m/s, max 5 cm/year
            vy_plume[j, i] = v_mag  # Rising

for step in range(100):
    result = thermal.solve_step(
        phase_field=phase_field,
        velocity_x=vx_plume,
        velocity_y=vy_plume,
        dt=1e12,
        use_advection=True  # SUPG stabilization active
    )
    
    if step % 20 == 0:
        # Plot temperature profile
        print(f"Step {step}: Plume top at y={plume_center[1]/1e3 + step*5e-9*1e12/1e3:.1f} km")
```

**Expected Behavior:**
```
Material rises carrying heat → steeper gradient
Hot material cools as it rises → thermal plume visible
Plume head spreads horizontally at top → mushroom shape
Energy balance: Advection >> Diffusion for plumes
```

### Heat Source (Radiogenic Heating)

```python
# Scenario: Radioactive element decay in crust
# Typical: 1-2 µW/m³ in crust

heat_generation = np.zeros((ny, nx))

# Add heat in crustal layer only
crustal_layer = (phase_field == 2)
heat_generation[crustal_layer] = 1e-6  # W/m³

# Solve with heating
for step in range(50):
    result = thermal.solve_step(
        phase_field=phase_field,
        heat_source=heat_generation,
        dt=1e12,
        use_advection=False
    )
    
    if step % 10 == 0:
        print(f"Step {step}: Crust heated by {result.temperature[50,50] - T_init[50,50]:.1f} K")
```

---

## 6. Performance Tuning

### Benchmarking Different Solvers

```python
from sister_py import benchmark_solver, OptimizedSolver

# Create test problem (Stokes)
A_stokes = create_stokes_matrix(nx, ny)  # Your assembly
b_stokes = np.random.random(A_stokes.shape[0])

# Benchmark all solvers
results = benchmark_solver(A_stokes, b_stokes)

print("Solver Performance Comparison:")
print(f"  Direct:     {results['direct']['time']:8.4f} s")
print(f"  GMRES:      {results['gmres']['time']:8.4f} s")
print(f"  BiCG-STAB:  {results['bicgstab']['time']:8.4f} s")
print(f"  Multigrid:  {results['multigrid']['time']:8.4f} s")
print()
print("Recommended: Multigrid")
```

### Memory Estimation

```python
from sister_py import estimate_memory_usage, estimate_flops

# For different problem sizes
for nx in [50, 100, 200, 500]:
    ny = nx
    n_dof = nx * ny
    
    mem_mb = estimate_memory_usage(n_dof)
    flops = estimate_flops(A_stokes, iterations=50)
    
    print(f"{nx:3d}×{ny:3d}: {mem_mb:6.1f} MB, {flops:.2e} FLOPS")
```

### Profiling Code

```python
from sister_py import profile_code

@profile_code()
def run_thermal_step(thermal_model, **kwargs):
    """Measure thermal solver time."""
    return thermal_model.solve_step(**kwargs)

# Run and get summary
for step in range(10):
    run_thermal_step(thermal, phase_field=phase_field, dt=1e12)

# Print summary
print(run_thermal_step.get_summary())
```

**Output:**
```
Thermal Solver Profile:
  Total time:     1.234 s
  Calls:          10
  Per-call avg:   0.123 s
  Min/Max:        0.110 / 0.145 s
```

---

## 7. Common Examples

### Example 1: Continental Rift (Isothermal)

```python
"""Simplified continental rifting - isothermal mechanics."""

import numpy as np
from sister_py import TimeIntegrator, RheologyModel

# Set up rift geometry
nx, ny = 200, 100
Lx, Ly = 500e3, 200e3
x = np.linspace(0, Lx, nx)
y = np.linspace(0, Ly, ny)

# Rheology: weak rift zone
viscosity = np.ones((ny, nx)) * 1e21
rift_zone = np.abs(x[None, :] - Lx/2) < 50e3  # ±50 km from center
viscosity[rift_zone, :] = 1e20  # 10× weaker

# Boundary conditions: pure shear (extension)
v_boundary = 1e-9  # 1 cm/year
vx = np.ones((ny, nx)) * v_boundary
vx[:, :nx//2] *= -1  # Left side pulls left

# Time stepping
integrator = TimeIntegrator(grid_x=x, grid_y=y, viscosity_field=viscosity)
dt = 1e12
for step in range(100):
    result = integrator.step(velocity_x=vx, velocity_y=np.zeros_like(vx), dt=dt)
    
    if step % 20 == 0:
        stress_max = np.max(np.abs(result.stress))
        print(f"Step {step:2d} ({step*dt/1e12:6.1f} My): σ_max = {stress_max/1e6:7.1f} MPa")

# Expected: Rift zone deforms faster, produces high stress
```

### Example 2: Subduction Zone (Coupled Thermo-Mechanical)

```python
"""Subducting slab with thermal structure."""

import numpy as np
from sister_py import TimeIntegrator, ThermalModel

# Slab geometry: cold, dipping plate
nx, ny = 200, 150
Lx, Ly = 500e3, 300e3
x = np.linspace(0, Lx, nx)
y = np.linspace(0, Ly, ny)
xx, yy = np.meshgrid(x, y)

# Temperature: slab is cold (500K), wedge is hot
T = 1600 * np.ones((ny, nx))  # Hot mantle
slab_zone = (xx < 200e3) & (yy < 100e3 - 0.2*xx)  # Dipping slab
T[slab_zone] = 500  # Cold slab

# Velocity: slab subducts at 5 cm/year
vx = np.zeros((ny, nx))
vy = np.zeros((ny, nx))
vx[slab_zone] = -0.05/1e6  # 5 cm/year into page
vy[slab_zone] = -0.05/1e6 * 0.2  # Dip angle

# Time stepping with thermal coupling
thermal = ThermalModel(grid_x=x, grid_y=y, initial_temperature=T)
for step in range(100):
    result = thermal.solve_step(
        phase_field=np.ones_like(T, dtype=int),
        velocity_x=vx, velocity_y=vy,
        dt=1e12, use_advection=True
    )
    T = result.temperature
    
    if step % 20 == 0:
        T_wedge = T[80:100, 150:170].mean()
        print(f"Step {step:2d}: Wedge T = {T_wedge:.0f} K")

# Expected: Slab remains cold, wedge warms with depth
```

### Example 3: Plume-Lithosphere Interaction

```python
"""Mantle plume impacting lithosphere."""

import numpy as np
from sister_py import ThermalModel, TimeIntegrator

nx, ny = 150, 200
Lx, Ly = 300e3, 400e3
x = np.linspace(0, Lx, nx)
y = np.linspace(0, Ly, ny)
xx, yy = np.meshgrid(x, y)

# Initial: plume head (hot) surrounded by cooler mantle
T_ambient = 1300  # K
T_plume = 1800    # K
r_plume = 50e3    # 50 km radius
plume_center = (Lx/2, Ly - 100e3)  # Plume head below surface

r = np.sqrt((xx - plume_center[0])**2 + (yy - plume_center[1])**2)
T = T_ambient + (T_plume - T_ambient) * np.exp(-(r/r_plume)**2)

# Velocity: plume rises
v_radius = 1e-8  # 1 cm/year radial velocity
vx = v_radius * (xx - plume_center[0]) / (r + 1e3)
vy = v_radius * (yy - plume_center[1]) / (r + 1e3)

# Time stepping
thermal = ThermalModel(grid_x=x, grid_y=y, initial_temperature=T)
for step in range(200):
    result = thermal.solve_step(
        phase_field=np.ones_like(T, dtype=int),
        velocity_x=vx, velocity_y=vy,
        dt=5e11,  # 5 ky per step for accuracy
        use_advection=True
    )
    T = result.temperature
    
    # Update velocity for rising plume
    r = np.sqrt((xx - plume_center[0])**2 + (yy - plume_center[1])**2)
    vx = v_radius * (xx - plume_center[0]) / (r + 1e3)
    vy = v_radius * (yy - plume_center[1]) / (r + 1e3)
    
    if step % 40 == 0:
        print(f"Step {step:3d}: Plume head at y = {plume_center[1]/1e3 + step*1e-8*1e6/1e3:.0f} km")

# Expected: Hot plume rises and spreads; mushroom head forms
```

---

## 8. Troubleshooting

### Problem: Solver Not Converging

**Symptom:**
```
Warning: GMRES did not converge (iterations=1000)
Solution may be inaccurate
```

**Solutions:**
```python
# 1. Reduce dt (lower CFL number)
dt = dt / 2  # Try half the time step

# 2. Switch solver manually
from sister_py import OptimizedSolver
solver = OptimizedSolver(method='multigrid')  # Force multigrid

# 3. Improve preconditioning
from sister_py import MultiGridPreconditioner
precond = MultiGridPreconditioner(n_levels=5)  # More levels

# 4. Check matrix conditioning
from scipy.sparse import linalg
cond_number = linalg.eigsh(A)[0][-1] / linalg.eigsh(A)[0][0]
print(f"Condition number: {cond_number:.2e}")
# If > 1e10: problem is ill-conditioned, needs preconditioning
```

### Problem: Temperature Oscillating/Non-Physical

**Symptom:**
```
Temperature: -500 K (physically impossible!)
Oscillating between 500 and 2000 K
```

**Solutions:**
```python
# 1. Check Peclet number
dx = x[1] - x[0]
dt = ...
v_max = 1e-8
Pe = v_max * dx * (rho * cp) / (2 * k)
print(f"Peclet number: {Pe:.2f}")
# If Pe > 1: use SUPG stabilization
use_advection = True

# 2. Reduce time step
dt = dt / 4

# 3. Check boundary conditions
print(f"T_min, T_max at each BC:")
print(f"  Top: {T[0, :].min():.0f}-{T[0, :].max():.0f}")
print(f"  Bottom: {T[-1, :].min():.0f}-{T[-1, :].max():.0f}")

# 4. Verify internal heating not too large
Q_internal_max = heat_source.max()
print(f"Max internal heat: {Q_internal_max:.2e} W/m³")
# Typical ranges: 0-2e-6 W/m³
```

### Problem: Markers Accumulating in Corners

**Symptom:**
```
Marker count: [100, 100, 2500, 50]  # Corner has 2500!
```

**Solutions:**
```python
# 1. Check marker reseeding
from sister_py import MarkerReseeding
reseeding = MarkerReseeding(
    target_markers_per_cell=100,
    reseed_threshold=50
)

# 2. Smooth velocity field (reduces clustering)
from scipy.ndimage import gaussian_filter
vx_smooth = gaussian_filter(vx, sigma=2)
vy_smooth = gaussian_filter(vy, sigma=2)

# 3. Use adaptive dt to prevent bunching
adaptive_dt = True
```

### Problem: Thermal Solver Hanging

**Symptom:**
```
Solving thermal equation... (program hangs for >10 min)
```

**Solutions:**
```python
# 1. Check matrix size
n_dof = nx * ny
print(f"Thermal DOFs: {n_dof}")
if n_dof > 1e6:
    print("Too large! Consider coarser grid")

# 2. Verify no NaN in inputs
assert np.all(np.isfinite(T_old))
assert np.all(np.isfinite(k_field))
assert np.all(np.isfinite(heat_source))

# 3. Force direct solver for debugging
from sister_py import OptimizedSolver
solver = OptimizedSolver(method='direct', verbose=True)
# Should complete in seconds for small problems

# 4. Add timeout
import signal
def timeout_handler(signum, frame):
    raise TimeoutError("Solver exceeded 60 seconds")

signal.signal(signal.SIGALRM, timeout_handler)
signal.alarm(60)  # 60 second limit
try:
    result = thermal.solve_step(...)
finally:
    signal.alarm(0)  # Cancel alarm
```

### Problem: Memory Running Out

**Symptom:**
```
MemoryError: Unable to allocate 8.2 GiB for an array
```

**Solutions:**
```python
# 1. Reduce mesh resolution
nx_new = nx // 2
ny_new = ny // 2
print(f"Memory reduction: {((nx*ny)/(nx_new*ny_new))**2:.1f}×")

# 2. Use single precision (float32) instead of float64
T = T.astype(np.float32)

# 3. Check for memory leaks
import gc
gc.collect()

# 4. Delete large temporary arrays
del T_history  # Don't store all time steps
```

---

## Quick Reference

### Common Constants (SI Units)
```python
# Physical constants
SECONDS_PER_YEAR = 365.25 * 24 * 3600
MILLION_YEARS = 1e6 * SECONDS_PER_YEAR

# Typical mantle parameters
MANTLE_VISCOSITY = 1e21  # Pa·s
MANTLE_DENSITY = 3300    # kg/m³
MANTLE_CONDUCTIVITY = 4  # W/m/K
MANTLE_CP = 1200         # J/kg/K
MANTLE_ALPHA = 2e-5      # 1/K

# Crustal parameters
CRUST_VISCOSITY = 1e20   # Pa·s (weaker due to water)
CRUST_DENSITY = 2800     # kg/m³
CRUST_CONDUCTIVITY = 2.5 # W/m/K

# Typical velocities
PLATE_VELOCITY = 5e-2 / SECONDS_PER_YEAR  # 5 cm/year in m/s
PLUME_VELOCITY = 10e-2 / SECONDS_PER_YEAR # 10 cm/year

# Typical time scales
AGE_OCEANIC_PLATE = 100 * MILLION_YEARS   # seconds
THERMAL_RELAXATION = 30 * MILLION_YEARS   # seconds
```

### Function Quick Start
```python
# Create and solve in 10 lines
from sister_py import TimeIntegrator, ThermalModel
import numpy as np

x, y = np.linspace(0,1e5,50), np.linspace(0,1e5,50)
T_init = 1500 - 10*(y[None,:]-y[:,None])
vx = 1e-9 * np.ones_like(T_init)
thermal = ThermalModel(x, y, T_init)

for step in range(10):
    result = thermal.solve_step(
        phase_field=np.ones_like(T_init, dtype=int),
        velocity_x=vx, velocity_y=np.zeros_like(vx), dt=1e12, use_advection=True
    )
    print(f"Step {step}: T_max={result.temperature.max():.0f}K")
```

