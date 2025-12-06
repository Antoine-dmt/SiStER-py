# Enhanced Speckit Prompts for SiSteR-py v2.0

**Production-Grade Geodynamic Code with Performance, Accuracy & Accessibility**

This document contains tuned Speckit prompts incorporating:
- ⚡ **Performance**: Optimization where it matters most
- 🎯 **Accuracy**: Fully-staggered grid (Duretz, May, Gerya 2013) for 30-50% error reduction
- 📦 **Distribution**: Pip-installable package ready (PyPI publishing path included)
- 🎓 **Accessibility**: Single YAML config file paradigm (like SiSteR MATLAB)

---

## Design Philosophy

### Key Principles for SiSteR-py v2.0

1. **Performance Where It Matters**
   - Focus optimization on: matrix assembly, interpolation, solver loops
   - Use Numba JIT for hot paths (marked `@njit` in code)
   - Defer GPU acceleration to Phase 5 (optional, not required)
   - Target: 1000 time steps in < 1 hour

2. **Fully-Staggered Grid for Accuracy**
   - All variables (P, Vx, Vy) on maximally separated node locations
   - Reduces discretization error by 30-50% vs standard staggered
   - Critical for variable viscosity (ratios > 10³)
   - Based on Arakawa C-grid principles

3. **Single-File Input Paradigm**
   - YAML config file (human-readable, unlike MATLAB .m)
   - One file per simulation (e.g., `continental_rift.yaml`)
   - Validation on load (clear error messages)
   - Example configs shipped with package

4. **Production-Ready Distribution**
   - pip-installable: `pip install sister-py`
   - Open source on GitHub/PyPI
   - Full CI/CD (pytest, coverage, build automation)
   - Package includes example inputs and data

### Implementation Phases

```
Phase 0: Setup (1 week)
  └─ Git, pyproject.toml, project structure, CI/CD

Phase 1: Config & Core Data (Weeks 2-3)
  ├─ ConfigurationManager (YAML loading)
  ├─ FullyStaggaredGrid (fully-staggered coordinates)
  ├─ Material & Rheology (modular physics)
  └─ Marker & MarkerSwarm (Lagrangian tracking)

Phase 2: Solvers (Weeks 4-6)
  ├─ Matrix Assembly (optimized FD, Numba-ready)
  ├─ Picard/Newton Solver (non-linear iteration)
  └─ Stress/Strain Rate (vectorized computation)

Phase 3: Integration (Weeks 7-8)
  ├─ Time Stepper (main loop orchestration)
  ├─ I/O System (HDF5, checkpointing)
  └─ Examples & Tutorials (continental rifting, etc.)

Phase 4: Distribution (Weeks 9-10)
  ├─ Package setup (setup.py, pyproject.toml)
  ├─ Documentation (API docs, installation guide)
  └─ Release pipeline (GitHub Actions, PyPI)

Phase 5: Optimization (Weeks 11+, optional)
  ├─ Profiling & bottleneck analysis
  ├─ Numba JIT compilation
  └─ GPU support (CuPy, optional)
```

---

## Prompt 0: Configuration System

### Prompt 0A: YAML-Based ConfigurationManager

```markdown
PROJECT_CONTEXT:
SiSteR-py follows the SiSteR MATLAB paradigm where a **single input file** 
drives the entire simulation. We use YAML (not MATLAB .m) for accessibility 
and maintainability. The ConfigurationManager must load, validate, and provide 
access to all simulation parameters from a single YAML file.

This component is CRITICAL for:
- Easy package distribution (example configs included)
- User accessibility (copy example YAML, modify parameters)
- Reproducibility (config exported with outputs)
- Extensibility (add new physics without code changes)

DESIGN_PRINCIPLES:
- Single-file input: `continental_rift.yaml` drives entire simulation
- YAML format (not JSON) for human readability and comments
- Validate all parameters on load (fail fast with clear messages)
- Support sensible defaults (users override what they need)
- Enable version control: config files tracked in git

SPECIFICATION:
Create `ConfigurationManager` class that:
1. Loads YAML configuration files (pyyaml)
2. Validates all parameters against schema (Pydantic v2)
3. Creates Material objects from config MATERIALS section
4. Creates boundary condition dict from config BC section
5. Creates grid config dict from config GRID section
6. Provides read-only access to all simulation parameters
7. Exports config to stdout/file for reproducibility

REQUIREMENTS:
- Load YAML files using pyyaml (handle multi-line strings, lists, dicts)
- Validation via Pydantic BaseModel classes:
  - SIMULATION: Nt (int > 0), dt_out (int > 0), output_dir (str)
  - DOMAIN: xsize (float > 0), ysize (float > 0)
  - GRID: spacing as piecewise {x: [dx values], x_breaks: [breakpoints]}
  - MATERIALS: list of {phase, density, rheology, plasticity, elasticity, thermal}
  - BC: {top, bot, left, right} each with type and value
  - PHYSICS: {elasticity: bool, plasticity: bool, thermal: bool}
  - SOLVER: {Npicard_min, Npicard_max, conv_tol, switch_to_newton}
- Error messages show WHICH parameter failed and WHY:
  - ❌ "friction at MATERIALS[1].plasticity.mu = 1.5, expected 0 < μ < 1"
  - ❌ "Grid spacing must be positive, got -500 at GRID.x_spacing[1]"
- Create material objects: Material(phase, density, rheology, etc.)
- Export methods:
  - `to_dict()`: return full config as nested dict
  - `to_yaml(filepath)`: save config to YAML file
  - `to_string()`: return formatted config as printable string
- Support includes:
  - Comments in YAML files (preserved for documentation)
  - Environment variable substitution: `${HOME}/data/file.nc`
  - Include other YAML files: `!include "base_config.yaml"`

CONSTRAINTS:
- YAML schema matches SiSteR MATLAB input file structure:
  ```yaml
  SIMULATION:
    Nt: 1600
    dt_out: 20
    output_dir: "./results"
  
  DOMAIN:
    xsize: 170e3
    ysize: 60e3
  
  GRID:
    x_spacing: [2000, 500, 2000]
    x_breaks: [50e3, 140e3]
    y_spacing: [2000, 500, 2000]
    y_breaks: [7e3, 40e3]
  
  MATERIALS:
    - phase: 1
      name: "Sticky Layer"
      density:
        rho0: 1000
        alpha: 0
      rheology:
        type: "ductile"
        diffusion: {A: 0.5e-18, E: 0, n: 1}
        dislocation: {A: 0.5e-18, E: 0, n: 1}
      plasticity: {C: 40e6, mu: 0.6}
      elasticity: {G: 1e18}
      thermal: {k: 3, cp: 1000}
    
    - phase: 2
      ...
  
  BC:
    top: {type: "velocity", vx: 1e-10, vy: 0}
    bot: {type: "velocity", vx: 0, vy: 0}
    left: {type: "velocity", vx: 1e-10, vy: 0}
    right: {type: "velocity", vx: 1e-10, vy: 0}
  
  PHYSICS:
    elasticity: true
    plasticity: true
    thermal: true
  
  SOLVER:
    Npicard_min: 10
    Npicard_max: 100
    conv_tol: 1e-9
    switch_to_newton: 0
  ```
- Use Pydantic v2 for validation (typed, composable)
- All validation errors collected (not just first, list ALL failures)
- Package includes example configs: `sister_py/data/examples/*.yaml`
- Default config: `sister_py/data/defaults.yaml` (hardcoded sensible values)

ACCESSIBILITY:
- Example YAML files well-documented with inline comments:
  ```yaml
  # Grid spacing: three zones with different resolutions
  # Zone 1: 0 to 50 km at 2000 m spacing
  # Zone 2: 50 to 140 km at 500 m spacing (fine resolution)
  # Zone 3: 140 to 170 km at 2000 m spacing
  x_spacing: [2000, 500, 2000]
  x_breaks: [50e3, 140e3]
  ```
- Validation messages suggest fixes:
  - "friction coefficient out of range: 1.5 > 1.0, try 0.3-0.6"
  - "Grid spacing must be > 0, got -500, check zone definitions"
- Quick start: copy example YAML from GitHub, modify 3-4 parameters
- Installation: `pip install sister-py` includes examples in `~/.sister_py/`
- Documentation: "5-Minute Quick Start" guide with YAML walkthrough

ACCEPTANCE_CRITERIA:
- Load `continental_rift.yaml` without errors (real MATLAB input converted)
- Reject invalid config with clear error:
  - μ=1.5 → "friction > 1.0, expected 0 < μ < 1"
  - viscosity_max < viscosity_min → "bounds reversed"
  - Grid spacing negative → lists which zone
- Round-trip: load YAML → modify param → save → load again (bit-identical config)
- Performance: load 1000-line config in < 100 ms
- Export: `config.to_yaml(file)` round-trips perfectly (6 sig figs)
- Create materials: `config.get_materials()` returns dict of Material objects

EXAMPLE_USAGE:
```python
from sister_py.config import ConfigurationManager

# Load config from YAML file
cfg = ConfigurationManager.load("continental_rift.yaml")

# Access nested parameters
print(cfg.DOMAIN.xsize)  # 170000.0
print(cfg.MATERIALS[0].density.rho0)  # 1000.0

# Get Material objects ready for simulation
materials = cfg.get_materials()
print(materials[1].phase)  # 2
print(materials[1].viscosity_ductile(...))  # Call rheology

# Modify parameters programmatically
cfg.SIMULATION.Nt = 100  # Override time steps
cfg.validate()  # Re-validate

# Export for reproducibility
cfg.to_yaml("my_run.yaml")
print(cfg.to_string())  # Print config to stdout
```
```

---

## Prompt 1: Fully-Staggered Grid

### Prompt 1A: FullyStaggaredGrid Class

```markdown
PROJECT_CONTEXT:
SiSteR-py uses a **fully-staggered grid** (Duretz, May, Gerya 2013) instead 
of the standard half-staggered approach used in original SiSteR. This is a 
critical design choice:

- **Why**: With variable viscosity (10¹⁸ to 10²⁵ Pa·s), fully-staggered grids 
  reduce discretization error by 30-50% vs half-staggered
- **How**: All variables (P, Vx, Vy) placed on maximally-separated node locations
- **Trade-off**: ~25% more memory, ~10% faster convergence → net win

The grid is the foundational data structure managing:
- Coordinates for pressure (P-nodes) and velocities (Vx/Vy-nodes)
- Index mappings for solution vector assembly
- Finite difference stencils (dx, dy, 1/dx, 1/dy)
- Interpolation between node types (conservative)

DESIGN_PRINCIPLES:
- Fully-staggered (Arakawa C-grid variant) for superior accuracy
- Optimize speed: grid creation < 10 ms for 1000×1000 cells
- All methods Numba-compatible (no Python objects in hot loops)
- Support variable grid spacing for mesh refinement
- Prepare all interpolation for JIT compilation (@njit compatible)

SPECIFICATION:
Create `FullyStaggaredGrid` class managing three staggered coordinate systems:

1. **P-nodes** (pressure): corner locations at (i·h, j·h)
   - Arranged in (Ny, Nx) array
   - Used for pressure values and cell-center quantities

2. **Vx-nodes** (horizontal velocity): offset locations at ((i+0.5)·h, j·h)
   - Arranged in (Ny, Nx+1) array
   - Used for v_x values and ∂v_x/∂x derivatives

3. **Vy-nodes** (vertical velocity): offset locations at (i·h, (j+0.5)·h)
   - Arranged in (Ny+1, Nx) array
   - Used for v_y values and ∂v_y/∂y derivatives

All three systems are **maximally separated** (no colocation).

REQUIREMENTS:
- Create three separate coordinate arrays:
  - `x_p` (shape Nx), `y_p` (shape Ny): pressure node locations
  - `x_vx` (shape Nx+1), `y_vx` (shape Ny): vx node locations
  - `x_vy` (shape Nx), `y_vy` (shape Ny+1): vy node locations
- Index mapping with three functions:
  - `index_p(i, j)` → linear index in pressure solution vector
  - `index_vx(i, j)` → linear index in vx solution vector
  - `index_vy(i, j)` → linear index in vy solution vector
  - Inverse functions: `coords_from_index(linear_idx, node_type)` → (i, j)
- Pre-compute FD stencil arrays (all marked for Numba):
  - `dx` (shape Nx): cell widths for ∂/∂x
  - `dy` (shape Ny): cell heights for ∂/∂y
  - `dxi` (shape Nx): 1/dx (avoid division in loops)
  - `dyi` (shape Ny): 1/dy
  - `dxxi` (shape Nx+1): 1/(2·dx) for second derivatives
  - `dyyi` (shape Ny+1): 1/(2·dy)
- Interpolation methods (4-point bilinear stencils):
  - `interp_p_to_vx(p_array)`: P-nodes → Vx-nodes (conservative averaging)
  - `interp_p_to_vy(p_array)`: P-nodes → Vy-nodes (conservative averaging)
  - `interp_vx_to_p(vx_array)`: Vx-nodes → P-nodes (conservative averaging)
  - `interp_vy_to_p(vy_array)`: Vy-nodes → P-nodes (conservative averaging)
  - All use weighted average (4 neighbors, weight by inverse distance)
- Validation: all spacings > 0, no gaps, coordinates monotonic increasing
- Memory layout: arrays in C-order (row-major) for Numba efficiency

CONSTRAINTS:
- Variable spacing format (matching SiSteR MATLAB input):
  ```python
  grid_config = {
      'xsize': 170e3,
      'ysize': 60e3,
      'x_spacing': [2000, 500, 2000],      # cell sizes in each zone
      'x_breaks': [50e3, 140e3],           # zone boundaries
      'y_spacing': [2000, 500, 2000],
      'y_breaks': [7e3, 40e3]
  }
  grid = FullyStaggaredGrid(grid_config)
  ```
- Solution vector layout: separate arrays for P, Vx, Vy (not interleaved)
  ```python
  p = solution[:Np]
  vx = solution[Np:Np+Nvx]
  vy = solution[Np+Nvx:]
  ```
- All coordinates in SI units (meters)
- Grid created once at init, immutable during simulation
- Interpolation uses bilinear (4-point) stencils for accuracy
- Mark all methods @njit-compatible (no Python-only constructs)

ACCESSIBILITY:
- Example grid configs in package: `sister_py/data/examples/grids/`
- Docstrings include ASCII diagrams showing node layout:
  ```
  j=0    j=1    j=2    (x-direction)
  
  i=0  P---P---P      ← P-nodes at corners
       |   |   |
       V---V---V      ← Vx-nodes (horizontal velocity)
       |   |   |
  i=1  P---P---P
       |   |   |
  ```
- Default grid (uniform) if spacing_config not provided
- Error messages: "Grid spacing must be > 0 at x_breaks[50e3]" not generic
- Tutorial: "Guide to Mesh Refinement" (how to create variable-spacing grids)

ACCEPTANCE_CRITERIA:
- Create grids: 50×25, 100×50, 200×100, 500×500 cells without error
- Index mapping round-trip: (i,j,type) → idx → (i,j,type) perfect recovery
- Grid properties correct:
  - P-nodes shape: (Ny, Nx)
  - Vx-nodes shape: (Ny, Nx+1)
  - Vy-nodes shape: (Ny+1, Nx)
  - Npred = Ny·Nx, Nvx = Ny·(Nx+1), Nvy = (Ny+1)·Nx
- Interpolation tests (with Numba compiled):
  - Constant field P=5.0 → interpolated Vx, Vy all ≈ 5.0 (exact)
  - Linear field P = 2·x + 3·y → Vx, Vy nodes exact to machine precision
  - Random field → all interpolated values within bounds, no NaN/Inf
  - Harmonic mean test: inverse averaging preserves harmonic behavior
- Variable spacing:
  - Grid transitions smoothly at breakpoints (no kinks in dx)
  - Spacing ratios smooth (|dx[i+1]/dx[i]| < 1.1 if possible)
- Performance:
  - Grid creation: 50×50 < 1ms, 500×500 < 10ms, 1000×1000 < 100ms
  - 10,000 interpolation calls: < 100 ms (Numba compiled)
- Memory usage: < 1 MB for 100×100 grid

EXAMPLE_USAGE:
```python
from sister_py.grid import FullyStaggaredGrid
from sister_py.config import ConfigurationManager
import numpy as np

# Load from config file
cfg = ConfigurationManager.load("continental_rift.yaml")
grid = FullyStaggaredGrid(cfg.GRID)

# Query grid properties
print(f"Pressure nodes: ({grid.Ny}, {grid.Nx})")  # (75, 150)
print(f"Vx nodes: ({grid.Ny}, {grid.Nx+1})")     # (75, 151)
print(f"Vy nodes: ({grid.Ny+1}, {grid.Nx})")     # (76, 150)

# Get linear indices
idx_p = grid.index_p(i=10, j=20)
idx_vx = grid.index_vx(i=10, j=20)
idx_vy = grid.index_vy(i=10, j=20)

# Interpolate field from P-nodes to Vx-nodes
p_vals = np.random.randn(grid.Ny, grid.Nx)
vx_vals = grid.interp_p_to_vx(p_vals)  # Shape (Ny, Nx+1)

# Get FD spacing arrays for matrix assembly
dx = grid.dx    # Shape (Nx,), cell widths
dy = grid.dy    # Shape (Ny,), cell heights
dxi = grid.dxi  # Shape (Nx,), 1/dx
dyi = grid.dyi  # Shape (Ny,), 1/dy

# Query node coordinates
x_at_p, y_at_p = grid.coord_p(i=5, j=10)
x_at_vx, y_at_vx = grid.coord_vx(i=5, j=10)

# Export grid metadata
grid.to_dict()  # Return grid info as dict
```
```

---

## Prompt 2: Material & Rheology System

### Prompt 2A: Material Class with Modular Rheology

```markdown
PROJECT_CONTEXT:
SiSteR represents rock properties via "phases" (e.g., "sticky layer", "mantle"). 
Each phase has multiple rheological models (ductile creep, plasticity, elasticity) 
that couple during simulation. In original SiSteR, rheology is scattered across 
many functions. We consolidate into modular `Material` and `Rheology` classes.

This component must:
- Store material parameters from config YAML
- Compute effective viscosity (ductile + plastic coupling)
- Support stress/strain rate dependency (critical for non-linear iteration)
- Be extensible (add new rheology models without changing existing code)

DESIGN_PRINCIPLES:
- **Modular rheology**: Each model (ductile, plastic, elastic) is separate class
- **Composition over inheritance**: Material composes rheology models
- **Mutable state**: Temperature, stress, strain rate evolve (not read-only)
- **Numba-ready**: All computations vectorizable (2D arrays, not nested objects)

SPECIFICATION:
Create:
1. `Rheology` (abstract base class)
   - `viscosity(sigma_II, strain_rate_II, T)` → returns η
   - `stress_evolution(stress, strain_rate, dt)` → returns updated stress
   
2. `DuctileRheology` (power-law creep)
   - Power law: ε̇ = A·σⁿ·exp(-E/nRT)
   - Supports diffusion + dislocation (harmonic mean)
   
3. `PlasticRheology` (Mohr-Coulomb yield)
   - Yield: σ_Y = (C + μ·P)·cos(arctan(μ))
   - Caps viscosity at σ_Y/(2·ε̇)
   
4. `ElasticRheology` (stress accumulation)
   - σ_elastic = 2·G·ε_elastic
   - Rotates with material flow
   
5. `Material` class
   - phase: int (phase ID, 1-indexed)
   - density: temperature-dependent ρ(T)
   - rheology_models: list of [DuctileRheology, PlasticRheology, ElasticRheology]
   - thermal: k, cp (conductivity, heat capacity)

REQUIREMENTS:
- Create Material from config dict:
  ```python
  cfg = ConfigurationManager.load("rift.yaml")
  materials = cfg.get_materials()  # list of Material objects
  ```
- Material properties (read/write):
  - `phase` (int, read-only)
  - `density(T)` method: ρ = ρ0·(1 - α·T)
  - `viscosity_ductile(sigma_II, eps_II, T)` → η_ductile
  - `viscosity_plastic(sigma_II, P)` → η_plastic (or ∞ if P < 0)
  - `viscosity_effective(sigma_II, eps_II, T, P)` → min(η_ductile, η_plastic)
- Parameter validation:
  - Creep n > 0, E ≥ 0, A > 0
  - Friction 0 < μ < 1
  - Cohesion C ≥ 0
  - Density ρ0 > 0, |α| < 0.1
  - Shear modulus G > 0
- Two-creep combination:
  - Harmonic mean for diffusion + dislocation coupling
  - η_combined = 1 / (1/η_diff + 1/η_disc)
- Export methods:
  - `to_dict()` → nested dict (config serializable)
  - `summary()` → string of material properties
- All methods marked @njit-compatible or @vectorize for NumPy broadcasting

CONSTRAINTS:
- SI units throughout: Pa, Pa·s, K, J/mol, s⁻¹
- Temperature in Kelvin (no Celsius)
- Activation energy E in J/mol, gas constant R = 8.314 J/(mol·K)
- Pre-exponential A has units [Pa^(-n)·s^(-1)]
- Stress σ in Pa, strain rate ε̇ in s⁻¹
- Density ρ in kg/m³
- Viscosity bounds: 10^18 to 10^25 Pa·s (enforced at assembly time)

ACCESSIBILITY:
- Example materials in package: `sister_py/data/examples/materials/`
- Clear parameter names in config:
  ```yaml
  rheology:
    ductile:
      diffusion:
        pre_exp: 0.5e-18  # A, pre-exponential
        activation_energy: 0  # E in J/mol
        stress_exponent: 1  # n
  ```
- Docstrings include typical values:
  - Viscosity at mid-lithosphere: 1e20-1e21 Pa·s
  - Activation energy (dislocation): 500-600 kJ/mol
  - Friction coefficient: 0.3-0.6
- Error messages suggest ranges:
  - "friction 1.5 out of range, try 0.3-0.6"

ACCEPTANCE_CRITERIA:
- Load materials from config: cfg.get_materials() returns list of Material
- Viscosity values match MATLAB SiSteR to 6 sig figs:
  - Ductile: T=1000K, ε̇=1e-15, σ=1e7 Pa → η ≈ 1e20 Pa·s
  - Plastic: C=40e6, μ=0.6, P=3e8 Pa, ε̇=1e-15 → η ≈ 5e20 Pa·s
  - Effective: min(η_ductile, η_plastic)
- Parameter validation:
  - Reject μ = -0.1 with error message
  - Reject n = 0 with error message
  - Reject G < 0 with error message
- Round-trip: Material → dict → YAML → load → Material (identical)
- Performance: compute viscosity for 100×100 grid in < 10 ms

EXAMPLE_USAGE:
```python
from sister_py.material import Material, DuctileRheology, PlasticRheology
from sister_py.config import ConfigurationManager

# Load from config
cfg = ConfigurationManager.load("rift.yaml")
mat_list = cfg.get_materials()
mantle = mat_list[1]  # phase 2

# Compute viscosity
sigma_II = 1e7  # Pa
eps_II = 1e-15  # /s
T = 1200  # K
P = 4e8  # Pa (pressure)

eta_ductile = mantle.viscosity_ductile(sigma_II, eps_II, T)
eta_plastic = mantle.viscosity_plastic(sigma_II, P)
eta_eff = mantle.viscosity_effective(sigma_II, eps_II, T, P)

# Temperature-dependent density
rho = mantle.density(T=1200)  # 3300 kg/m³ × (1 - α·ΔT)

# Export material properties
print(mantle.summary())
mantle_dict = mantle.to_dict()
```
```

---

## Prompt 3: Marker System

### Prompt 3A: MarkerSwarm with Vectorized Operations

```markdown
PROJECT_CONTEXT:
Lagrangian markers track material properties (phase, composition, stress, 
strain history) through time as they advect with flow. Original SiSteR uses 
parallel MATLAB arrays (xm, ym, sxxm, etc.) - error-prone and hard to extend.

SiSteR-py uses `Marker` (semantic wrapper) and `MarkerSwarm` (vectorized array 
operations). This component must:
- Advect markers with velocity field (CFL-limited)
- Store stress/strain history
- Interpolate to/from grid (bilinear)
- Reseed to maintain density
- Support efficient HDF5 save/load

DESIGN_PRINCIPLES:
- **Vectorized operations**: Use NumPy arrays, not Python loops
- **Numba-ready**: All hot loops marked @vectorize or @njit
- **Lazy interpolation**: Don't compute until needed (memory efficient)
- **Immutable after init**: Marker properties don't change (only positions/stress)

SPECIFICATION:
Create:
1. `Marker` (thin wrapper, semantic clarity)
   - Properties: x, y, phase, sxx, sxy, T, e_p
   - Methods: none (just data container)

2. `MarkerSwarm` (main class, vectorized operations)
   - Storage: NumPy arrays for x, y, phase, stress, temperature, strain
   - Methods:
     - `__init__(n_markers, grid)`: initialize uniform distribution
     - `advect(vel_x_grid, vel_y_grid, dt)`: move markers (bilinear interp)
     - `update_stress(strain_rate_grid, dt, materials)`: evolve stresses
     - `interpolate_to_nodes(grid, property)`: scatter to grid
     - `interpolate_from_nodes(grid, nodal_field)`: gather from grid
     - `reseed(grid, min_density, materials)`: maintain marker density
     - `save(filepath)`: HDF5 export
     - `load(filepath)`: HDF5 import
     - `copy()`: deep copy for checkpointing
     - `remove_outside(grid)`: remove markers outside domain

REQUIREMENTS:
- Initialize swarm with uniform distribution:
  - num_per_cell markers in each grid cell
  - Total ~num_per_cell² × Nx × Ny markers
  - Positions: random within cell (not exactly at centers)
- Marker array layout (all NumPy, C-order):
  ```python
  xm = np.ndarray((n_markers,), dtype=np.float64)  # positions
  ym = np.ndarray((n_markers,), dtype=np.float64)
  im = np.ndarray((n_markers,), dtype=np.int32)    # phase
  sxxm = np.ndarray((n_markers,), dtype=np.float64)  # deviatoric stress
  sxym = np.ndarray((n_markers,), dtype=np.float64)
  ep = np.ndarray((n_markers,), dtype=np.float64)   # plastic strain
  Tm = np.ndarray((n_markers,), dtype=np.float64)  # temperature
  ```
- Advection via bilinear interpolation of velocities:
  - For each marker: v_marker = bilinear_interp(v_grid, x_marker, y_marker)
  - New position: x_new = x_old + v_x·dt (with CFL check)
  - Must preserve marker order (no sorting)
- Stress update from strain rate:
  - ε̇ from grid → interpolated to markers
  - σ_new = σ_old + Δσ(ε̇, dt, material, T)
  - Handle elasticity: σ = σ_old + 2G·Δε_viscous - σ·ω·dt
- Interpolation to grid:
  - Scatter marker properties to nodes (weighted by marker position)
  - Handle variable cell sizes (volume weighting)
  - Support: phase, density, temperature, stress components
- Interpolation from grid:
  - Gather nodal values to marker locations
  - Bilinear interpolation (4-point stencil)
  - Support all field types: scalar (T), vector (v), tensor (σ)
- Reseeding strategy:
  - Count markers in each cell
  - If density < min, add new markers uniformly in cell
  - New markers inherit phase, stress from neighbors (interpolation)
  - If density > 2×max, optionally remove excess
- I/O:
  - Save to HDF5: `/markers/x`, `/markers/y`, `/markers/phase`, etc.
  - Load and recreate exactly
  - Include metadata: n_markers, time, simulation id

CONSTRAINTS:
- Time step limited by CFL: dt < 0.5·min(dx,dy)/max(|v|)
- All array operations NumPy-compatible (Numba-ready)
- No nested Python objects (Numba can't JIT)
- Marker indexing must be stable (don't sort, preserve id)
- Temperature in Kelvin, stresses in Pa, position in meters
- Mark hot loops @njit compatible

ACCESSIBILITY:
- Example: "Initializing marker swarm" in documentation
- Default: 10 markers per cell (user can override in config)
- Docstrings show typical marker counts:
  - Small domain (100×50 km): ~50,000 markers
  - Medium domain (170×60 km): ~100,000 markers
  - Large domain (500×300 km): ~2M markers

ACCEPTANCE_CRITERIA:
- Initialize 100,000 marker swarm: < 500 ms
- Advect all markers: v=1 m/s constant, Δt=100 s → displacement=100 m (verify)
- Interpolate ε̇ from grid to markers: verify with analytical function
- Reseed: remove cluster of 1000 markers, reseed, new markers appear
- Save/load roundtrip: 50,000 markers → HDF5 → reload → checksums match

EXAMPLE_USAGE:
```python
from sister_py.marker import MarkerSwarm
from sister_py.grid import FullyStaggaredGrid

# Initialize swarm
grid = FullyStaggaredGrid(config.GRID)
swarm = MarkerSwarm(num_per_cell=10, grid=grid)
swarm.initialize_phases(config.GEOMETRY)

# Advect and update
swarm.advect(vx_grid, vy_grid, dt)
swarm.update_stress(strain_rate_grid, dt, materials)

# Interpolate to grid
phase_on_nodes = swarm.interpolate_to_nodes(grid, property='phase')
rho_on_nodes = swarm.interpolate_to_nodes(grid, property='density')

# Reseed if needed
swarm.reseed(grid, min_density=5, materials=materials)

# Checkpoint
swarm.save("checkpoint_iter500.h5")
swarm_new = MarkerSwarm.load("checkpoint_iter500.h5")
```
```

---

## Prompt 4: Matrix Assembly

### Prompt 4A: OptimizedStokesMatrixAssembler

```markdown
PROJECT_CONTEXT:
The core of SiSteR: assemble sparse FD matrix L and RHS vector R for the 
linear system L·S = R (Stokes equations on staggered grid). This is the 
computational bottleneck - must be fast but correct.

Original SiSteR assembles in MATLAB (slow). We optimize in Python:
- Vectorize where possible (NumPy broadcasting)
- Numba JIT on tight loops
- Target: < 100 ms for 100×100 grid assembly

DESIGN_PRINCIPLES:
- **Performance focus**: Hot loop (assembly) marked for JIT
- **Sparse matrix**: Use scipy.sparse (COO → CSR for solving)
- **Correctness first**: Validate against MATLAB outputs
- **Fully-staggered ready**: Use grid from FullyStaggaredGrid

SPECIFICATION:
Create `OptimizedStokesMatrixAssembler` class:
1. Takes grid, viscosity, density, BC → assembles L, R
2. Returns scipy.sparse CSR matrix + dense vector
3. Handles all BCs: Dirichlet velocity, Neumann stress, periodic
4. Scales equations (Kc, Kb) for numerical stability
5. Supports elasticity: adds stress history to RHS

REQUIREMENTS:
- FD stencil assembly:
  - Momentum equations (3 components each per point)
  - Continuity equation (1 per pressure node)
  - 4-point compact stencil for efficiency
- Matrix scaling via Kc, Kb coefficients
- Boundary condition handling:
  - Dirichlet: replace equation row with v = value
  - Neumann: add to RHS
  - Periodic: wrap indices
  - Pressure anchor: fix one P node
- Sparse format: scipy.sparse.csr_matrix (efficient solve)
- RHS includes gravity: ρ·g terms
- Return: (L, R, Kc, Kb) tuple
- Support elasticity: srhs_xx, srhs_xy in RHS

CONSTRAINTS:
- Fully-staggered grid indexing from FullyStaggaredGrid
- Picard iteration loop calls this ~10-100 times per time step
- Must be deterministic (same inputs → same outputs)
- All operations Numba-compatible (hot loop annotated)

ACCESSIBILITY:
- Example: "Assembling Stokes system" in documentation
- Test case: simple channel flow with known analytical solution
- Profile output: time spent in assembly, assembly calls per iteration

ACCEPTANCE_CRITERIA:
- Assemble 100×100 grid: < 50 ms (100 ms with validation)
- Matrix properties: (3Np+Nvx+Nvy) × (3Np+Nvx+Nvy), sparsity < 2%
- Symmetry check: L ≈ L^T (saddle-point structure)
- Test: analytical solution comparison (plane strain, linear BCs)
- 10 Picard iterations on 50×50 grid: < 2 seconds total

EXAMPLE_USAGE:
```python
from sister_py.solver import OptimizedStokesMatrixAssembler
import numpy as np

assembler = OptimizedStokesMatrixAssembler(grid)

etas = np.ones((grid.Ny, grid.Nx)) * 1e20
etan = np.ones((grid.Ny+1, grid.Nx+1)) * 1e20
rho = np.ones_like(etas) * 3300
BC = {
    'top': {'type': 'velocity', 'value': (1e-10, 0)},
    'bot': {'type': 'velocity', 'value': (0, 0)},
}

L, R, Kc, Kb = assembler.assemble(etas, etan, rho, BC)

# Solve
from scipy.sparse.linalg import spsolve
S = spsolve(L, R)
p, vx, vy = grid.unpack_solution(S)
```
```

---

## Prompt 5: Non-Linear Solver

### Prompt 5A: StokesNonlinearSolver with Adaptive Switching

```markdown
PROJECT_CONTEXT:
Non-linear iteration: viscosity depends on stress/strain rate, so we can't 
solve once. Use:
1. Picard iterations (S_new = L^{-1}·R) - robust, slow
2. Newton iterations (S_new = S - L^{-1}·Res) - fast, needs good guess

SiSteR-py adapts: start Picard, switch to Newton mid-solve.

DESIGN_PRINCIPLES:
- **Adaptive strategy**: Picard → Newton automatically
- **Robust**: Works from bad initial guesses
- **Fast**: Converges in 10-50 iterations typical
- **Observable**: Print convergence history

SPECIFICATION:
`StokesNonlinearSolver` class:
1. Manages Picard/Newton iteration loop
2. Calls MatrixAssembler each iteration
3. Checks convergence (residual norm)
4. Switches strategy adaptively
5. Returns: converged (bool), S, n_iterations, residuals

REQUIREMENTS:
- Constructor params: max_iter, tol, switch_iter, verbose
- Main method: `solve(viscosity_init, rheology, materials, BC)`
  - Picard for iterations 1 to switch_iter
  - Newton for iterations switch_iter+1 to max_iter
  - Convergence check: ||LS - R||_2 / ||R||_2 < tol
- Return dict: {'converged': bool, 'solution': S, 'n_iter': int, 'residuals': array}
- Track residuals (for plotting)
- Adaptive switch logic: if residual plateaus, switch early

CONSTRAINTS:
- Picard: S_new = L^{-1}·R (full solve)
- Newton: S_new = S - L^{-1}·(L·S - R) (uses previous S)
- Iterative solver with preconditioner (for large grids)
- Handle singular L (pressure nullspace): anchor pressure

ACCESSIBILITY:
- Default params: 100 max iters, tol=1e-9, switch at iter 0 (pure Newton)
- Easy to debug: residuals printed each iteration (if verbose=True)

ACCEPTANCE_CRITERIA:
- Converge in < 50 iterations for test cases
- Picard: linear convergence, Newton: superlinear
- Example plot: residuals vs iteration (L2 norm)

EXAMPLE_USAGE:
```python
solver = StokesNonlinearSolver(max_iter=100, tol=1e-9, switch_iter=10)
result = solver.solve(viscosity_init, rheology_models, materials, BC)
print(f"Converged: {result['converged']}")
print(f"Iterations: {result['n_iter']}")
# Plot: plt.semilogy(result['residuals'])
```
```

---

## Prompt 6: Time Stepper & Main Loop

### Prompt 6A: GeodynamicsSimulation with Configuration-Driven Workflow

```markdown
PROJECT_CONTEXT:
Main time loop orchestrating: solve → advect → update stresses → output. 
Configuration-driven (ConfigurationManager loads all params).

SPECIFICATION:
`GeodynamicsSimulation` class:
1. Load config (YAML file)
2. Initialize grid, materials, markers, solver
3. Main loop: step through time
4. Output on demand, checkpointing

REQUIREMENTS:
- Constructor: takes ConfigurationManager
- Methods:
  - `run(output_dir)`: main loop
  - `step()`: single time iteration
  - `get_state()`: return current solution
  - `save_checkpoint(filepath)`: HDF5
  - `load_checkpoint(filepath)`: restore
- Output: HDF5 with fields (v, p, T, viscosity, phase, stresses)

CONSTRAINTS:
- Time loop follows SiSteR sequence (11 steps)
- Adaptive time stepping (CFL)
- Optional physics flags (elasticity, plasticity, thermal)

ACCESSIBILITY:
- Example: `GeodynamicsSimulation(cfg).run("results/")`
- Example configs provided
- Tutorial: "First simulation" (5 minutes)

ACCEPTANCE_CRITERIA:
- Run 10 time steps without error
- Markers reseed correctly
- Output files created
- Can checkpoint and resume

EXAMPLE_USAGE:
```python
from sister_py.simulation import GeodynamicsSimulation
from sister_py.config import ConfigurationManager

cfg = ConfigurationManager.load("continental_rift.yaml")
sim = GeodynamicsSimulation(cfg)
sim.run(output_dir="results/")

# Or step-through
for i in range(100):
    sim.step()
    if i % 10 == 0:
        sim.save_checkpoint(f"checkpoint_{i:04d}.h5")
```
```

---

## Final Notes on Accessibility & Distribution

### Package Installation

```bash
# Install from PyPI
pip install sister-py

# Install from source (development)
git clone https://github.com/user/sister-py.git
cd sister-py
pip install -e .
```

### Project Structure for Distribution

```
sister-py/
├── pyproject.toml              # Package metadata
├── setup.py                     # Legacy, for compatibility
├── README.md                    # Installation, quick start
├── LICENSE                      # Open source (MIT or GPL)
│
├── sister_py/                   # Package source
│   ├── __init__.py
│   ├── config.py               # ConfigurationManager
│   ├── grid.py                 # FullyStaggaredGrid
│   ├── material.py             # Material, Rheology classes
│   ├── marker.py               # Marker, MarkerSwarm
│   ├── solver.py               # Solvers (assembly, Picard/Newton)
│   ├── simulation.py           # GeodynamicsSimulation, main loop
│   ├── io.py                   # HDF5 I/O
│   │
│   └── data/
│       ├── examples/           # Example YAML configs
│       │   ├── continental_rift.yaml
│       │   ├── subduction.yaml
│       │   └── shear_flow.yaml
│       │
│       └── defaults.yaml       # Default parameters
│
├── tests/                       # Unit & integration tests
│   ├── test_grid.py
│   ├── test_material.py
│   ├── test_marker.py
│   ├── test_solver.py
│   └── test_integration.py
│
├── examples/                    # Runnable examples
│   ├── continental_rifting.py
│   └── benchmark.py
│
├── docs/                        # Documentation
│   ├── index.md
│   ├── installation.md
│   ├── quick_start.md          # "Run in 5 minutes"
│   ├── configuration.md
│   ├── api_reference.md
│   └── physics.md
│
└── .github/
    └── workflows/
        ├── pytest.yml          # Run tests on push
        └── publish.yml         # Publish to PyPI
```

### Accessibility Best Practices

1. **Installation**: One-liner `pip install sister-py`
2. **Getting Started**: Copy example YAML from `~/.sister_py/examples/`
3. **Documentation**: Full API docs + tutorials + physics explanations
4. **Examples**: Runnable scripts in `examples/` folder
5. **Community**: GitHub issues, discussions, contributing guide

---

## Execution Checklist

When you're ready for each phase, follow this order:

**Phase 0 Setup (Week 1)**
- [ ] Initialize Git repo, pyproject.toml
- [ ] Set up pytest framework
- [ ] Refine Prompt 0A (ConfigurationManager)

**Phase 1 Core Data (Weeks 2-3)**
- [ ] Use Prompt 0A → build ConfigurationManager
- [ ] Use Prompt 1A → build FullyStaggaredGrid
- [ ] Use Prompt 2A → build Material & Rheology
- [ ] Use Prompt 3A → build MarkerSwarm

**Phase 2 Solvers (Weeks 4-6)**
- [ ] Use Prompt 4A → build OptimizedStokesMatrixAssembler
- [ ] Use Prompt 5A → build StokesNonlinearSolver
- [ ] Validate against MATLAB test cases

**Phase 3 Integration (Weeks 7-8)**
- [ ] Use Prompt 6A → build GeodynamicsSimulation
- [ ] Integrate all components
- [ ] Run continental rifting example

**Phase 4 Distribution (Weeks 9-10)**
- [ ] Polish package structure
- [ ] Write full documentation
- [ ] Set up CI/CD (GitHub Actions)
- [ ] Publish to PyPI

**Phase 5 Optimization (Weeks 11+)**
- [ ] Profile code
- [ ] Apply Numba JIT to hot loops
- [ ] Optional: GPU support with CuPy

---

**You now have production-grade Speckit prompts ready for implementation! 🚀**
