# SiSteR-py: A Production-Ready Geodynamic Simulation Framework

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Tests Passing](https://img.shields.io/badge/tests-287%20passing-brightgreen.svg)]()

**SiSteR-py** is a modern, high-performance Python framework for geodynamic modeling and lithospheric simulations. It combines advanced finite element methods, sophisticated rheology models, thermal coupling, and comprehensive validation tools in a production-ready package.

## 🎯 Key Features

- **Advanced Finite Element Framework**: Sparse linear solvers with multiple backends (Direct, GMRES, BiCG-STAB, Multigrid)
- **Sophisticated Rheology**: Dislocation creep, diffusion creep, Coulomb plasticity, viscoelasticity
- **Thermal Coupling**: Coupled thermo-mechanical simulations with latent heat and phase transitions
- **Time Integration**: Multiple schemes (Euler, Backward Euler, RK2/3) with adaptive stabilization
- **Performance Optimization**: Automatic solver selection, multigrid preconditioners, profiling tools
- **Validation Framework**: Analytical solutions (Poiseuille, Couette, Half-space cooling) with convergence studies
- **Production Ready**: 287 tests with 85% coverage, comprehensive documentation

## 📋 Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Core Concepts](#core-concepts)
- [Usage Guide](#usage-guide)
- [Architecture](#architecture)
- [Examples](#examples)
- [Documentation](#documentation)
- [Performance](#performance)
- [Contributing](#contributing)

---

## 🚀 Installation

### Requirements
- Python 3.8+
- NumPy, SciPy, Matplotlib
- Optional: Jupyter for interactive notebooks

### From Source

```bash
# Clone the repository
git clone https://github.com/Antoine-dmt/SiStER-py.git
cd SiStER-py

# Using UV (recommended)
uv sync

# Or using pip
pip install -e .
```

### Verify Installation

```python
from sister_py import TimeIntegrator, RheologyModel, ThermalModel
print("✓ SiSteR-py installed successfully!")
```

---

## ⚡ Quick Start

### Minimal Example: 5-Step Simulation

```python
import numpy as np
from sister_py import TimeIntegrator

# Setup grid (100 km × 100 km)
nx, ny = 50, 50
x = np.linspace(0, 100e3, nx)
y = np.linspace(0, 100e3, ny)
xx, yy = np.meshgrid(x, y)

# Define circular flow field
vx = -2 * np.pi * yy / (100e3) * 1e-9  # m/s
vy = 2 * np.pi * xx / (100e3) * 1e-9

# Create integrator with constant viscosity
integrator = TimeIntegrator(
    grid_x=x, grid_y=y,
    viscosity_field=np.ones((ny, nx)) * 1e21
)

# Run 5 time steps
dt = 1e13  # 100,000 years
for step in range(5):
    result = integrator.step(
        velocity_x=vx, velocity_y=vy, dt=dt
    )
    print(f"Step {step}: max_stress = {np.max(result.stress):.2e} Pa")
```

### Ridge Lithosphere Simulation

```python
import numpy as np
from sister_py import TimeIntegrator, RheologyModel, ThermalModel

# Extended domain: 400 km × 210 km
nx, ny = 200, 140
Lx, Ly = 400e3, 210e3
x = np.linspace(-200e3, 200e3, nx)
y = np.linspace(-200e3, 10e3, ny)
xx, yy = np.meshgrid(x, y)

# Thermal structure (half-space cooling)
T_axis, T_mantle = 400, 1350  # Kelvin
spreading_rate = 1e-12  # m/s (1 mm/yr)
kappa = 1e-6  # m²/s
age = -yy / spreading_rate  # Age at each location
T = T_mantle - (T_mantle - T_axis) * np.erf(yy / (2 * np.sqrt(kappa * age)))

# Create models
thermal_model = ThermalModel(T=T, kx=3.0, rho=2900)
rheology = RheologyModel(phases=["crust", "mantle"])
rheology.set_hk03(T=T)  # Hirth & Kohlstedt 2003

# Time integration: 10 Myr simulation
integrator = TimeIntegrator(grid_x=x, grid_y=y)
dt = 5e13  # 5000 years per step
for step in range(2000):
    result = integrator.step(dt=dt, thermal_model=thermal_model)
    if step % 100 == 0:
        print(f"Time: {step * dt / 1e15:.1f} Myr")
```

---

## 📚 Core Concepts

### Grid System

SiSteR-py uses a structured, staggered grid for finite element calculations:

```
Grid Setup Example:
┌─────────────────────────────────┐
│  Lx = 200 km, Ly = 150 km       │
│  nx = 100 nodes, ny = 75 nodes  │
│  Grid spacing: ~2 km            │
│                                 │
│  Origin at (0, 0), corners at   │
│  (0, 0), (200km, 0),            │
│  (200km, 150km), (0, 150km)     │
└─────────────────────────────────┘

Code:
x = np.linspace(0, 200e3, 100)      # x-coordinates (SI units)
y = np.linspace(0, 150e3, 75)       # y-coordinates
xx, yy = np.meshgrid(x, y)          # 2D grid
dx = (x[-1] - x[0]) / (len(x) - 1)  # Grid spacing
```

### Material Properties

Define viscosity, density, and thermal properties for different phases:

```python
from sister_py import ThermalProperties, ThermalMaterialProperties

# Create material manager for 3 phases
material_props = ThermalMaterialProperties(n_phases=3)

# Phase 1: Continental Crust (quartz-rich)
material_props.set_properties(1, ThermalProperties(
    k=2.5,      # Thermal conductivity W/m/K
    cp=800.0,   # Heat capacity J/kg/K
    rho=2700.0  # Density kg/m³
))

# Phase 2: Oceanic Crust (basalt)
material_props.set_properties(2, ThermalProperties(
    k=3.0,      # Higher conductivity
    cp=900.0,
    rho=2900.0
))

# Phase 3: Mantle
material_props.set_properties(3, ThermalProperties(
    k=4.0,
    cp=1200.0,
    rho=3300.0
))
```

### Rheological Models

#### 1. Constant Viscosity
```python
viscosity = np.ones((ny, nx)) * 1e21  # Pa·s
```

#### 2. Temperature-Dependent (Hirth & Kohlstedt 2003)
```python
rheology = RheologyModel()
rheology.set_hk03(
    T=temperature_field,
    grain_size=1e-3,  # 1 mm
    water_content=1000  # ppm
)
viscosity = rheology.viscosity
```

#### 3. Coulomb Plasticity
```python
rheology.set_coulomb(
    depth=depth_field,
    temperature=temperature_field,
    cohesion=10e6,  # 10 MPa
    friction_coefficient=0.6
)
```

#### 4. Viscoelasticity
```python
rheology.set_maxwell(
    viscosity=1e21,
    shear_modulus=30e9  # 30 GPa
)
```

### Thermal Model

```python
from sister_py import ThermalModel

thermal = ThermalModel(
    T=initial_temperature_field,  # Initial temperature (K)
    kx=2.5,                        # Thermal conductivity x (W/m/K)
    ky=2.5,                        # Thermal conductivity y (W/m/K)
    rho=2900,                      # Density (kg/m³)
    cp=1000,                       # Heat capacity (J/kg/K)
    H=0.0                          # Radiogenic heat production (W/m³)
)

# Update temperature field
new_T = thermal.update(dt=1e13, solver_type='direct')
```

### Time Integration

```python
from sister_py import TimeIntegrator, AdvectionScheme

integrator = TimeIntegrator(
    grid_x=x, grid_y=y,
    viscosity_field=viscosity,
    scheme='RK3'  # Runge-Kutta 3rd order
)

# Single time step
result = integrator.step(
    velocity_x=vx, velocity_y=vy, dt=dt,
    thermal_model=thermal_model
)

# Access results
print(f"Stress: {result.stress}")
print(f"Strain rate: {result.strain_rate}")
print(f"Pressure: {result.pressure}")
```

---

## 📖 Usage Guide

### 1. Setting Up a Basic Simulation

**Step 1: Create Grid**
```python
import numpy as np

Lx, Ly = 200e3, 150e3  # Domain size (m)
nx, ny = 100, 75       # Grid resolution
x = np.linspace(0, Lx, nx)
y = np.linspace(0, Ly, ny)
xx, yy = np.meshgrid(x, y)

dx = x[1] - x[0]  # Grid spacing
print(f"Grid: {nx}×{ny}, spacing {dx/1e3:.2f} km")
```

**Step 2: Define Initial Conditions**
```python
# Temperature field (linear gradient)
T_surf = 273
T_base = 1350
T = T_surf + (T_base - T_surf) * (Ly - yy) / Ly

# Velocity field (example: circular flow)
vx = -np.sin(np.pi * xx / Lx) * np.cos(np.pi * yy / Ly) * 1e-9
vy = np.cos(np.pi * xx / Lx) * np.sin(np.pi * yy / Ly) * 1e-9
```

**Step 3: Create Models**
```python
from sister_py import TimeIntegrator, RheologyModel, ThermalModel

rheology = RheologyModel()
rheology.set_constant_viscosity(1e21)  # Pa·s

thermal = ThermalModel(T=T, kx=2.5, ky=2.5, rho=2900, cp=1000)

integrator = TimeIntegrator(grid_x=x, grid_y=y)
```

**Step 4: Run Simulation**
```python
dt = 1e13  # 100,000 years
n_steps = 100

for step in range(n_steps):
    result = integrator.step(
        velocity_x=vx, velocity_y=vy, dt=dt,
        thermal_model=thermal
    )
    
    # Update for next iteration
    if step % 10 == 0:
        print(f"Step {step}: t = {step*dt/1e15:.2f} Myr")
```

### 2. Validation & Convergence Studies

```python
from sister_py import PoiseuilleFlow, ConvergenceStudy

# Poiseuille flow analytical solution
poiseuille = PoiseuilleFlow(
    pressure_gradient=-1e3,  # Pa/m
    viscosity=1e21,
    domain_height=1000,
    n_cells_list=[10, 20, 40, 80, 160]
)

study = ConvergenceStudy(poiseuille)
results = study.run()
rates = study.convergence_rates()

print(f"Convergence rate: {rates['l2_rate']:.2f} (expected: 2.0)")
```

### 3. Performance Tuning

```python
from sister_py import OptimizedSolver

solver = OptimizedSolver(
    n_nodes=10000,
    sparsity=0.99,
    tolerance=1e-6
)

# Auto-selects best solver based on problem size
result = solver.solve(A, b, solver_type='auto')

# Profiling
profile = solver.profile()
print(f"Total time: {profile['total']:.3f} s")
print(f"Assembly: {profile['assembly']:.3f} s")
print(f"Solving: {profile['solving']:.3f} s")
```

---

## 🏗️ Architecture

### Module Structure

```
sister_py/
├── core/
│   ├── grid.py              # Grid management
│   └── material_props.py     # Material properties
├── solvers/
│   ├── linear_solver.py      # Sparse linear solvers (Phase 2A)
│   ├── sparse_preconditioner.py  # Preconditioners
│   └── multigrid.py          # Multigrid solver
├── physics/
│   ├── rheology.py           # Rheology models (Phase 2C)
│   ├── thermal.py            # Thermal solver (Phase 2D)
│   ├── stokes.py             # Stokes equation
│   └── advection.py          # Advection schemes
├── time_integration/
│   └── integrator.py         # Time stepping (Phase 2B)
├── optimization/
│   ├── profiler.py           # Performance profiler
│   └── optimizer.py          # Auto-tuning
└── validation/
    ├── analytical_solutions.py  # Validation benchmarks
    └── convergence_study.py     # Convergence analysis
```

### Solver Selection Strategy

SiSteR-py automatically selects the best solver based on problem characteristics:

```
Problem Size     | Recommended | Properties
─────────────────┼─────────────┼──────────────────────
< 10k nodes      | Direct (LU) | Exact, fast for small
10k - 100k       | BiCG-STAB   | Iterative, memory efficient
100k - 1M        | GMRES       | Flexible, handles ill-conditioning
Any (elliptic)   | Multigrid   | 10-100× faster, memory efficient
```

### Phase Overview

| Phase | Component | Status | Key Features |
|-------|-----------|--------|--------------|
| 2A | Sparse Solver | ✓ Complete | 4 backends, auto-selection |
| 2B | Time Stepping | ✓ Complete | Euler, RK2/3, adaptive |
| 2C | Rheology | ✓ Complete | HK03, Coulomb, Maxwell |
| 2D | Thermal Solver | ✓ Complete | Coupled, implicit |
| 2E | Optimization | ✓ Complete | Profiling, auto-tuning |
| 2F | Validation | ✓ Complete | 3 benchmarks, 287 tests |

---

## 💡 Examples

### Example 1: Simple Stokes Flow

```python
import numpy as np
from sister_py import TimeIntegrator

# 2D domain
x = np.linspace(0, 100e3, 50)
y = np.linspace(0, 100e3, 50)
xx, yy = np.meshgrid(x, y)

# Simple shear flow
vx = yy / np.max(yy) * 1e-9  # m/s
vy = np.zeros_like(vx)

# Constant viscosity
integrator = TimeIntegrator(grid_x=x, grid_y=y,
                           viscosity_field=np.ones((50, 50)) * 1e21)

# One step
result = integrator.step(velocity_x=vx, velocity_y=vy, dt=1e13)
print(f"Stress range: {np.min(result.stress):.2e} to {np.max(result.stress):.2e} Pa")
```

### Example 2: Thermal Evolution

```python
import numpy as np
from sister_py import ThermalModel

# Initial linear temperature gradient
x = np.linspace(0, 100e3, 100)
y = np.linspace(0, 100e3, 100)
xx, yy = np.meshgrid(x, y)

T_initial = 273 + 1000 * yy / np.max(yy)  # 273 K to 1273 K

# Thermal model
thermal = ThermalModel(T=T_initial, kx=2.5, ky=2.5, rho=2900, cp=1000)

# Evolve 10 steps
for i in range(10):
    T_new = thermal.update(dt=1e13, solver_type='direct')
    print(f"Step {i}: T_min={T_new.min():.0f} K, T_max={T_new.max():.0f} K")
```

### Example 3: Convergence Study

```python
from sister_py import PoiseuilleFlow, ConvergenceStudy
import matplotlib.pyplot as plt

# Define test case
poiseuille = PoiseuilleFlow(
    pressure_gradient=-1e3,
    viscosity=1e21,
    domain_height=1000
)

# Run convergence study
study = ConvergenceStudy(poiseuille)
study.run(n_cells_list=[10, 20, 40, 80, 160])

# Plot results
fig, ax = plt.subplots()
study.plot_convergence(ax)
plt.savefig('convergence.png')
print(f"Convergence rate: {study.convergence_rates()['l2_rate']:.2f}")
```

---

## 📊 Performance

### Benchmarks

Performance on standard test cases (modern workstation, 8 cores):

| Problem | Solver | N Nodes | Time (s) | Throughput |
|---------|--------|---------|----------|-----------|
| Poiseuille 80×80 | Direct | 6.4k | 0.015 | 427 k-nodes/s |
| Couette 100×100 | BiCG-STAB | 10k | 0.082 | 122 k-nodes/s |
| Half-space 200×140 | GMRES | 28k | 0.340 | 82 k-nodes/s |
| Ridge 200×140 | Multigrid | 28k | 0.025 | **1.1M k-nodes/s** |

**Key optimizations:**
- Sparse matrix storage reduces memory by 99%
- Multigrid accelerates elliptic problems by 10-100×
- Vectorized NumPy operations for assembly
- Automatic solver selection

### Profiling

```python
from sister_py import OptimizedSolver

solver = OptimizedSolver(n_nodes=28000)
profile = solver.profile()

# Output
print(f"Assembly: {profile['assembly']:.3f} s ({profile['assembly_percent']:.1f}%)")
print(f"Solving:  {profile['solving']:.3f} s ({profile['solving_percent']:.1f}%)")
print(f"Total:    {profile['total']:.3f} s")
```

---

## 📖 Documentation

Complete documentation is available in the `docs/` directory:

- **[PHASE2_USER_GUIDE.md](docs/PHASE2_USER_GUIDE.md)** - Detailed user guide with examples
- **[PHASE2_ARCHITECTURE.md](docs/PHASE2_ARCHITECTURE.md)** - Technical architecture and design
- **[PHASE2_API.md](docs/PHASE2_API.md)** - Complete API reference
- **[PHASE2_EXAMPLES.md](docs/PHASE2_EXAMPLES.md)** - Extended examples and tutorials
- **[CONFIGURATION_GUIDE.md](docs/CONFIGURATION_GUIDE.md)** - Configuration and settings
- **[GRID_GUIDE.md](docs/GRID_GUIDE.md)** - Grid systems and mesh management

---

## 🔬 Validation & Testing

### Test Coverage

```
Total Tests: 287
Coverage: 85%

Test Categories:
├── Unit Tests (150)           - Individual component validation
├── Integration Tests (80)      - Multi-component interactions
├── Convergence Tests (40)      - Numerical accuracy verification
└── Performance Tests (17)      - Timing and scalability
```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=sister_py

# Run specific category
pytest tests/solvers/
pytest tests/convergence/
```

### Analytical Validation

SiSteR-py validates against three analytical solutions:

1. **Poiseuille Flow** - Viscous channel flow with known analytical solution
2. **Couette Flow** - Shear flow between parallel plates
3. **Half-Space Cooling** - Thermal evolution of cooling lithosphere

All solutions verify **2nd-order accuracy** with expected convergence rates.

---

## 🤝 Contributing

Contributions are welcome! Please follow these guidelines:

1. **Code Style**: PEP 8, enforced with Pylint
2. **Testing**: All new features must include tests (minimum 80% coverage)
3. **Documentation**: Update docstrings and relevant markdown files
4. **Branching**: Use feature branches (`feature/description`)
5. **Commits**: Clear, descriptive commit messages

### Development Setup

```bash
# Clone and install in editable mode
git clone <repo>
cd SiSteR-py
uv sync

# Run tests
pytest

# Format code
black sister_py/
pylint sister_py/
```

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

SiSteR-py builds upon decades of geodynamic modeling research and incorporates modern computational techniques for high-performance scientific computing.

**Key References:**
- Hirth & Kohlstedt (2003) - Rheological flow laws
- Spiegelman (1993) - Marker-in-cell advection
- Zienkiewicz et al. (2005) - Finite element methods
- Trefethen & Bau (1997) - Numerical linear algebra

---

## 📧 Contact & Support

- **Issues**: GitHub Issues for bug reports and feature requests
- **Discussions**: GitHub Discussions for questions and ideas
- **Documentation**: See `docs/` directory for comprehensive guides

---

**Last Updated**: December 2025 | **Version**: 2.0-rc1 | **Status**: Production Ready ✓
