# Speckit Prompt Design Guide for SiSteR-py

**Enhanced Design for Production-Grade Geodynamic Code**

This document provides strategies for designing effective Speckit prompts to build a **high-performance, accessible, production-ready** Python/OOP version of SiSteR. 

Key design principles for SiSteR-py v2.0:
- ⚡ **Performance**: Optimize where it matters (solvers, interpolation)
- 🎯 **Accuracy**: Fully-staggered grid for reduced discretization error
- 📦 **Distribution**: Pip-installable package + online repository
- 🎓 **Accessibility**: Single input file paradigm (YAML/JSON config)

---

## Part 1: Speckit Fundamentals for SiSteR-py

### What is Speckit Good For in This Context?

Speckit excels at:
- **Scaffolding**: Building modular, testable components
- **Code generation**: Creating well-structured boilerplate with clear patterns
- **Interface design**: Defining clean contracts between components
- **Documentation**: Generating comprehensive docstrings, type hints, tests
- **Optimization**: Generating optimized paths with Numba/GPU annotations ready

### SiSteR-py Development Workflow

```
Phase 0: Project Setup
  ├─ Git repo, pyproject.toml, package structure
  └─ CI/CD pipeline (pytest, coverage, build)
  
Phase 1: Core Data Structures (Weeks 1-2)
  ├─ FullyStaggaredGrid (improved numerical accuracy)
  ├─ Material & Rheology (modular physics)
  └─ Marker & MarkerSwarm (vectorized Lagrangian tracking)
  
Phase 2: Solver Core (Weeks 3-5)
  ├─ Optimized Matrix Assembly (performance-focused)
  ├─ Non-linear Solver (Picard→Newton with adaptive switching)
  └─ Stress/Strain Rate Computation (Numba-ready)
  
Phase 3: Integration & Config (Weeks 6-8)
  ├─ Configuration System (YAML input files, extensible)
  ├─ Time Stepper (main simulation loop)
  └─ I/O & Checkpointing (HDF5 for interoperability)
  
Phase 4: Distribution (Weeks 9-10)
  ├─ Package setup.py/pyproject.toml
  ├─ Documentation (API docs, tutorials)
  └─ GitHub/PyPI integration (pip install sister-py)
  
Phase 5: Optimization (Weeks 11+)
  ├─ Profile & identify bottlenecks
  ├─ Numba JIT compilation (hot loops)
  └─ GPU acceleration (CuPy, optional)
```

---

## Part 2: Enhanced Prompt Template for SiSteR-py

### 2.1 Anatomy of an Effective SiSteR-py Speckit Prompt

```
PROJECT_CONTEXT:
[Brief overview of SiSteR-py and this component's role in the architecture]

DESIGN_PRINCIPLES:
- [Principle 1: e.g., "Optimize for speed where profiling shows bottlenecks"]
- [Principle 2: e.g., "Use fully-staggered grid for accuracy"]
- [Principle 3: e.g., "Prepare all code for Numba JIT compilation"]

SPECIFICATION:
[Clear definition of what needs to be built, with reference to SiSteR MATLAB]

REQUIREMENTS:
- [Functional requirement 1]
- [Functional requirement 2]
- [Performance requirement: e.g., "assembly of 100×100 grid < 10ms"]
- [API requirement: e.g., "match scipy.sparse interface"]
- [Documentation requirement: e.g., "type hints for all functions"]

CONSTRAINTS:
- [Technical constraint: e.g., "NumPy arrays only, Numba-compatible"]
- [Dependency: e.g., "depends on StokesGrid from Phase 1"]
- [Convention: e.g., "SI units throughout, docstring format"]

ACCESSIBILITY:
- [How does this component support easy package distribution?]
- [Does this work with YAML/JSON config files?]
- [Can users understand how to use this from examples?]

ACCEPTANCE_CRITERIA:
- [Verifiable criterion 1: numerical test with tolerance]
- [Verifiable criterion 2: performance benchmark]
- [Verifiable criterion 3: API compatibility test]

EXAMPLE_USAGE:
[Show how this module will be used in practice, with realistic example]

OPTIMIZATION_NOTES:
- [Known bottleneck: "matrix assembly loops"]
- [Proposed solution: "vectorize with NumPy, Numba JIT ready"]
- [Profiling target: "measure before and after optimization"]
```

---

## Part 3: Prompt 0 - Configuration & Package Management

### Prompt 0A: Configuration System (YAML-Based Input)

```markdown
PROJECT_CONTEXT:
SiSteR-py follows the SiSteR MATLAB paradigm where a single input file drives the 
entire simulation. Unlike MATLAB's .m scripts, we'll use YAML for accessibility and 
parsing. This config system must:
- Load from a single file (e.g., `continental_rift.yaml`)
- Validate all parameters before simulation starts
- Be easily extended for new physics/materials
- Support both beginner-friendly and expert-level parameters
- Integrate seamlessly with the package distribution

DESIGN_PRINCIPLES:
- Single-file input paradigm (like SiSteR_Input_File_continental_rift.m)
- YAML for human readability (better than JSON for this use case)
- Pydantic models for validation and type safety
- Support sensible defaults (users override only what they need)
- Package includes example configs in `sister_py/data/examples/`

SPECIFICATION:
Create a `ConfigurationManager` class that:
1. Loads YAML config files with validation
2. Stores all simulation parameters (domain, materials, solver, physics flags)
3. Provides runtime checks (grid size > 0, viscosity bounds, etc.)
4. Exports config to stdout/file for reproducibility
5. Supports hierarchical defaults (global → case-specific)

REQUIREMENTS:
- Load YAML files (use pyyaml library)
- Validate types and ranges:
  - Grid spacing: positive, non-zero
  - Viscosity: 10^18 to 10^25 Pa·s
  - Friction coefficient: 0 < μ < 1
  - Temperature: K (absolute)
- Error messages show which parameter failed and why
- Export: dict() and to_yaml() methods
- Support includes: scalar + vectorized grid definitions
- Create materials from config (instantiate Material objects)
- Create boundary conditions from config (instantiate BC dict)
- Create marker swarm config from config

CONSTRAINTS:
- YAML file schema must match SiSteR MATLAB input file structure:
  ```yaml
  SIMULATION:
    Nt: 1600
    dt_out: 20
  DOMAIN:
    xsize: 170e3
    ysize: 60e3
  GRID:
    spacing:
      x: [2000, 500, 2000]
      x_breaks: [50e3, 140e3]
      y: [2000, 500, 2000]
      y_breaks: [7e3, 40e3]
  MATERIALS:
    - phase: 1
      density: {rho0: 1000, alpha: 0}
      rheology: {type: "ductile", ...}
  ...
  ```
- Use Pydantic v2 for validation
- All validation errors list ALL failed parameters (not just first)
- Package includes: `sister_py/data/examples/continental_rift.yaml` template

ACCESSIBILITY:
- Example config files are well-commented
- Validation errors suggest fixes ("friction > 1, expected 0 < μ < 1")
- Users can copy example YAML and modify parameters
- Package installation puts example files at `~/.sister_py/examples/`
- Documentation: "Configuration Guide" in main README

ACCEPTANCE_CRITERIA:
- Load `continental_rift.yaml` without errors
- Reject invalid config: μ=1.5 → clear error message
- Reject invalid config: viscosity_max < viscosity_min → clear error
- Round-trip: load → modify param → save → load again (identical)
- Performance: load 1000-line config in < 100ms
- Export: config to YAML matches input to 6 significant figures

EXAMPLE_USAGE:
```python
from sister_py.config import ConfigurationManager

# Load from file
cfg = ConfigurationManager.load("continental_rift.yaml")

# Validate (automatic on load)
cfg.validate()  # raises ValueError if invalid

# Access parameters
print(cfg.DOMAIN.xsize)  # 170e3
print(cfg.MATERIALS[0].rheology.type)  # "ductile"

# Modify parameters programmatically
cfg.SIMULATION.Nt = 100  # override time steps
cfg.validate()  # re-validate

# Export for reproducibility
with open("my_run.yaml", "w") as f:
    cfg.save(f)
```
```

---

## Part 4: Prompt 1 - Fully-Staggered Grid Implementation


### Prompt 1A: StokesGrid Class

```markdown
CONTEXT:
SiSteR-py is a Python rewrite of a MATLAB geodynamic code. The grid 
is the foundational data structure - it manages coordinates, spacing,
and index mappings for a staggered (MAC) grid used in solving the 
Stokes equations. Currently, grid creation is scattered across multiple 
MATLAB functions (SiStER_initialize_grid, index calculations in matrix 
assembly).

SPECIFICATION:
Create a `StokesGrid` class that encapsulates all grid-related operations.
The grid uses a staggered layout where pressure nodes are at corners,
velocity nodes at cell edges. Support variable grid spacing (refined 
regions) via piecewise-linear spacing definitions.

REQUIREMENTS:
- Support 2D Cartesian coordinates with variable spacing
- Separate coordinate systems: normal nodes (pressure), shear nodes (velocity)
- Create index mappings: global 3D solution vector ↔ (i,j) grid coordinates
- Support bilinear interpolation between node types
- Pre-compute grid spacing arrays (dx, dy, dxi, dyi) for FD stencils
- Validate grid consistency (dx > 0, dy > 0, no gaps)

CONSTRAINTS:
- Follow NumPy conventions: arrays are row-major, indexing is (i,y) then (j,x)
- Variable spacing defined by breakpoints (like SiSteR input format):
  GRID.dx = [500, 1000], GRID.x = [50e3, 170e3]
  means: 500 m spacing from x=0 to 50 km, 1000 m from 50 to 170 km
- Index ordering in solution vector: [p(0,0), vx(0,0), vy(0,0), p(0,1), ...]
- All coordinates in SI units (meters)

ACCEPTANCE CRITERIA:
- Generate grids 100×50, 200×100 cells correctly
- Index mapping: (i,j) → linear index and back, verified for corners, 
  center, edges
- Interpolation: test with constant field (should give same value everywhere),
  linear field (should be exact), random field (should be smooth)
- Variable spacing: test that mesh transitions smoothly at breakpoints
- Performance: grid creation < 10 ms for 1000×1000 cells

EXAMPLE USAGE:
```python
grid_config = {
    'xsize': 170e3,
    'ysize': 60e3,
    'spacing': {
        'x': ([2000, 500, 2000], [50e3, 140e3]),  # dx, x_breaks
        'y': ([2000, 500, 2000], [7e3, 40e3])
    }
}
grid = StokesGrid(grid_config)

# Access coordinates
x_normal = grid.x_normal  # Shape (Nx,)
y_normal = grid.y_normal  # Shape (Ny,)

# Map between node types
p_nodes = grid.interpolate_normal_to_shear(p_values)  # (Nx+1)×(Ny+1) → Nx×Ny

# Get FD stencil spacings for matrix assembly
dxi = grid.dxi  # 1/dx for each cell
dyi = grid.dyi  # 1/dy for each cell
```
```

**Comment**: This prompt is **concrete** - it specifies the exact input format 
(matching MATLAB), output format, and numerical acceptance criteria.

### Prompt 1B: Material & Phase System

```markdown
CONTEXT:
SiSteR represents materials via "phases" (e.g., "sticky layer" = phase 1,
"mantle" = phase 2). Each phase has rheological properties (viscosity laws),
density, thermal properties. Currently scattered across material property
files and arrays.

SPECIFICATION:
Create a `Phase` enum and `Material` class to encapsulate:
- Phase identification (integer or enum)
- Density model: ρ = ρ0(1 - α·T)
- Ductile rheology: power-law creep parameters (A, E, n for diffusion/dislocation)
- Plasticity: Mohr-Coulomb (cohesion C, friction μ)
- Elasticity: shear modulus G
- Thermal properties: thermal conductivity k, heat capacity cp

REQUIREMENTS:
- Store parameters as attributes with clear naming (no single-letter abbreviations)
- Support creation from dict (for YAML/JSON config loading)
- Validate parameter ranges (e.g., n > 0, 0 < μ < 1, G > 0)
- Two creep types per phase: diffusion + dislocation (combined harmonically)
- Export parameters to pandas DataFrame for visualization/logging
- Method: `viscosity_ductile(sigma, eps_rate, T)` returns ductile viscosity
- Method: `viscosity_plastic(sigma, P)` returns plastic (yield-limited) viscosity
- Method: `density(T)` returns temperature-dependent density

CONSTRAINTS:
- Viscosity calculations use SI units: Pa·s
- Stress in Pa, strain rate in 1/s, T in K
- Activation energy E in J/mol, R = 8.314 J/(mol·K)
- Pre-exponential A in dimensionally consistent units (A^-1/n has [Pa·s])
- All attributes read-only after creation (use @property, no setters)

ACCEPTANCE CRITERIA:
- Instantiate material from MATLAB input (MAT struct) → Python Material
- Viscosity values match MATLAB code to 6 significant figures for test cases:
  - T=1000 K, ε̇=1e-15 /s, σ=1e7 Pa → check both phases
- Plastic viscosity capped correctly: η_plas = σ_Y / (2·ε̇)
- Parameter validation: reject μ=-0.1, reject n=0, etc.
- DataFrame export roundtrip: Material → dict → Material (no loss)

EXAMPLE USAGE:
```python
phases = {
    'sticky_layer': Material(
        phase=1,
        density_model={'rho0': 1000, 'alpha': 0},
        ductile_creep={'A_diff': 0.5/1e18, 'E_diff': 0, 'n_diff': 1,
                       'A_disc': 0.5/1e18, 'E_disc': 0, 'n_disc': 1},
        plasticity={'C': 40e6, 'mu': 0.6},
        elasticity={'G': 1e18},
        thermal={'k': 3, 'cp': 1000}
    ),
    'mantle': Material(...)
}

# Compute effective viscosity
sigma_II = 1e7  # Pa
eps_II = 1e-15  # 1/s
T = 1000  # K
eta = phases['mantle'].viscosity_ductile(sigma_II, eps_II, T)
```
```

---

## Part 4: Phase 2 - Markers & Interpolation

### Prompt 2A: Marker Class & MarkerSwarm

```markdown
CONTEXT:
Lagrangian markers track material properties (phase, composition, stress, 
strain history) through time. In MATLAB, they're stored as parallel arrays
(xm, ym, im, sxxm, etc.). This is error-prone: easy to forget to update
all arrays, hard to extend. Need OOP encapsulation.

SPECIFICATION:
Create:
1. `Marker` class: Single particle with position, phase, stress, strain history
2. `MarkerSwarm` class: Collection of markers with vectorized operations

REQUIREMENTS (Marker):
- Position: x, y (float)
- Phase: integer identifying material
- Stresses: σ_xx, σ_xy (elastic storage)
- Plastic strain: e_p (cumulative)
- Temperature: T (advected)
- Velocity: v_x, v_y (interpolated from grid, cached)
- Unique ID for tracking
- All as properties with validation (T > 0, etc.)

REQUIREMENTS (MarkerSwarm):
- Initialize uniform distribution: n markers per grid cell
- Methods for batch operations:
  - `advect(velocity_field, dt)`: update positions
  - `update_stress(strain_rate, dt, material)`: stress evolution
  - `get_phase()`: return phase array for all markers
  - `interpolate_to_nodes(grid, quantity)`: weighted average to nodal grid
  - `interpolate_from_nodes(grid, nodal_field)`: scatter nodal values to markers
- Reseed: remove markers outside domain, add where density low
- Query: get markers in region, by phase, by stress state, etc.
- Persistence: save/load to HDF5

CONSTRAINTS:
- Use NumPy arrays internally (xm, ym as 1D arrays, not lists of Marker objects)
  for performance
- Marker class is thin wrapper for semantic clarity; most operations on SwarmArray
- Advection uses bilinear interpolation from velocity field
- Interpolation to nodes uses inverse distance / volume weighting

ACCEPTANCE CRITERIA:
- Create swarm of 10,000 markers: < 1 ms
- Advect all: u=1 m/s constant, Δt=1000 s → displacement = 1 km, verified
- Interpolate ε̇ from nodes to markers: verify using analytical function
  (e.g., linear field should be exact)
- Reseed: manually remove a cluster, reseed, verify new markers added
- Save/load roundtrip: 50,000 markers → HDF5 → reload → checksums match

EXAMPLE USAGE:
```python
swarm = MarkerSwarm(grid, num_per_cell=10)
swarm.initialize_phases(geometry)

# Time step
swarm.advect(velocity_grid, dt)
swarm.update_stress(strain_rate_grid, dt, materials[swarm.phase])

# Interpolate to nodes for assembly
phase_on_nodes = swarm.interpolate_to_nodes(grid, 'phase')
rho_on_nodes = swarm.interpolate_to_nodes(grid, 'density')

# Check density
print(f"Total markers: {len(swarm)}")
print(f"Markers in phase 1: {(swarm.phase == 1).sum()}")

# Reseed after advection
swarm.reseed(grid, min_density=5)
```
```

---

## Part 5: Phase 3 - Stokes Solver

### Prompt 3A: System Matrix Assembly

```markdown
CONTEXT:
The heart of SiSteR: assemble the sparse FD matrix L and RHS vector R for:
  L·S = R  where S = [p, vx, vy] stacked, one per grid point.

This is complex: 659 lines of MATLAB (SiStER_assemble_L_R). Need to:
- Discretize momentum equations on staggered grid
- Apply boundary conditions (velocity, stress, periodic)
- Handle variable viscosity (non-uniform grid)
- Scale properly to avoid ill-conditioning

SPECIFICATION:
Create `StokesMatrixAssembler` class that:
1. Takes grid, viscosity fields (etas, etan), density, boundary conditions
2. Loops through grid points, assembles momentum + continuity equations
3. Returns sparse L matrix and RHS vector R
4. Supports different BC types: Dirichlet (velocity), Neumann (stress), periodic

REQUIREMENTS:
- FD stencil for momentum (4-point stencil: ∂σ/∂x, ∂σ/∂y)
- FD stencil for continuity (divergence of velocity)
- Scaling: compute Kc (momentum scale), Kb (continuity scale) from viscosity
- Boundary handling:
  - Top/bottom/left/right: specify velocity or stress BC
  - Pressure anchor: fix one P value to remove nullspace
  - Corners: handled smoothly (no singular points)
- Variable viscosity: use etas at shear nodes, etan at normal nodes
- Body forces: ρg in RHS
- Elasticity support: add deviatoric stress history term to RHS (srhs_xx, srhs_xy)
- Output: scipy.sparse.csr_matrix (efficient for solving)

CONSTRAINTS:
- Matrix indexing: row/col for (3 components × Nx × Ny), ordered [p, vx, vy] per point
- Boundary conditions defined by dict:
  ```
  BC = {
    'top': {'type': 'velocity', 'value': [ux, uy]},
    'bot': {'type': 'velocity', 'value': [ux, uy]},
    'left': {'type': 'stress', 'value': [tx, ty]},
    'right': {'type': 'stress', 'value': [tx, ty]}
  }
  ```
- Must match MATLAB scaling behavior (Kc, Kb definitions)

ACCEPTANCE CRITERIA:
- Matrix shape: (3·Nx·Ny) × (3·Nx·Ny)
- Matrix is symmetric (LS system should be)
- Test case 1: uniform viscosity, zero BC → L invertible, solution is zero (up to pressure)
- Test case 2: analytical solution (e.g., plane strain, linear velocity BC)
  → solve LS, compare to analytical
- Performance: assemble 100×100 grid with 10 Picard iterations: < 1 second total
- Sparsity: matrix should be < 5% dense

EXAMPLE USAGE:
```python
assembler = StokesMatrixAssembler(grid)

# Set up problem
etas = np.ones((grid.Nx, grid.Ny)) * 1e20  # shear viscosity
etan = np.ones((grid.Nx+1, grid.Ny+1)) * 1e20  # normal viscosity
rho = np.ones_like(etas) * 3300

BC = {
    'top': {'type': 'velocity', 'value': (1, 0)},  # extension
    'bot': {'type': 'velocity', 'value': (0, 0)},  # fixed
    'left': {'type': 'velocity', 'value': (1, 0)},
    'right': {'type': 'velocity', 'value': (1, 0)}
}

# Assemble
L, R, Kc, Kb = assembler.assemble(etas, etan, rho, BC)

# Solve
from scipy.sparse.linalg import spsolve
S = spsolve(L, R)
p, vx, vy = grid.unpack_solution(S)
```
```

### Prompt 3B: Non-Linear Solver (Picard/Newton)

```markdown
CONTEXT:
Viscosity depends on stress/strain rate → system is non-linear. SiSteR 
uses Picard iterations (robust) → switches to Newton (faster). Need to 
encapsulate this iteration strategy.

SPECIFICATION:
Create `StokesNonlinearSolver` class that:
1. Given current stress/strain rate, compute rheological viscosity
2. Assemble L, R via MatrixAssembler
3. Solve linear system
4. Extract strain rate, stress, residual
5. Check convergence
6. Switch strategy (Picard → Newton) if configured

REQUIREMENTS:
- Constructor: max_iterations, convergence_tolerance, switch_iteration
- Main loop: `solve(initial_viscosity, rheology_model, max_picard)`
  - Pit = 1 to switch_iteration: Picard update S_new = L^{-1}R
  - Pit > switch_iteration: Newton update S_new = S - (L^{-1}(LS - R))
  - Convergence check: ||LS - R||_2 / ||R||_2 < tolerance
  - Return: converged (bool), S, num_iterations
- Restart capability: start from existing S or from scratch
- Verbosity control: print residual every N iterations or silent

CONSTRAINTS:
- Picard iteration: S_new = L^{-1}R (full solve)
- Newton iteration: needs previous S (requires initialization)
- Viscosity updated each iteration from current ε̇ or σ
- Must handle near-singular systems (iterative solver with preconditioner)
- Residual calculation: Res = L·S - R, L2 norm

ACCEPTANCE CRITERIA:
- Converge to tol=1e-9 in < 50 iterations for test case
- Picard converges linearly, Newton superlinear (test with residuals)
- Restart from poor initial guess: compare to fresh start (should reach same solution)
- Singular L (no gravity, no BC pressure) → residual saturates at null-space level
- Performance: 10 iterations on 100×100 grid: < 5 seconds

EXAMPLE USAGE:
```python
solver = StokesNonlinearSolver(
    grid=grid,
    matrix_assembler=assembler,
    max_iterations=100,
    convergence_tol=1e-9,
    picard_switch=10,  # iterations before Newton
    verbose=True
)

# Estimate initial viscosity (constant)
eta_init = 1e20 * np.ones_like(etas)

# Solve
result = solver.solve(
    viscosity=eta_init,
    strain_rate_init=np.ones_like(etas) * 1e-15,
    materials=materials,
    BC=BC
)

print(f"Converged: {result['converged']}")
print(f"Iterations: {result['n_iter']}")
print(f"Final residual: {result['residual_final']}")

p = result['pressure']
vx = result['velocity_x']
vy = result['velocity_y']
```
```

---

## Part 6: Phase 4 - Time Integration

### Prompt 4A: Time Stepper & Main Loop

```markdown
CONTEXT:
SiSteR_MAIN.m orchestrates: solve → advect → update stresses → output.
Currently just a big for-loop with many function calls. Need OOP design
that's extensible and testable.

SPECIFICATION:
Create `GeodynamicsSimulation` class that encapsulates the entire time loop:

REQUIREMENTS:
1. Constructor takes:
   - Grid, materials, initial markers, BC
   - Configuration dict (Nt, dt_out, solver_params, rheology flags)
2. Methods:
   - `step()`: single time iteration (solve + advect + update)
   - `run(n_steps)`: loop and save outputs
   - `get_state()`: return current (p, vx, vy, T, phase, stress)
   - `set_state()`: restore from checkpoint
3. Checkpoint I/O:
   - Save: time, solution, markers, parameters
   - Load: resume from any iteration
4. Adaptive time stepping:
   - Compute dt from CFL condition: dt < 0.5 * min(dx, dy) / max(|v|)
5. Output:
   - Every dt_out iterations
   - Save fields: v, p, T, viscosity, strain rate, stress, phase

CONSTRAINTS:
- Time loop sequence (from SiStER_MAIN):
  1. Compute material properties at nodes from markers
  2. Solve Stokes (non-linear)
  3. Interpolate strain rate to markers
  4. Update marker stresses
  5. Update plastic strain
  6. OUTPUT (if requested)
  7. Set adaptive time step
  8. Rotate elastic stresses (if enabled)
  9. Diffuse temperature (if enabled)
  10. Advect markers, reseed
  11. Update topography markers
- Flags for optional physics (elasticity, plasticity, thermal)
- All times in SI units (seconds)

ACCEPTANCE CRITERIA:
- Run 10 steps of continental rifting model: completes without error
- Output files created: iteration 1, 5, 10
- Energy stability: check if LS·residual / ||R|| ~ 0 (conservation)
- Marker count: reseed maintains ~constant number (verify ranges)
- Checkpoint: save at step 5, stop, load, continue → identical step 10 as without stopping

EXAMPLE USAGE:
```python
config = {
    'Nt': 100,
    'dt_out': 10,
    'solver': {'max_iter': 100, 'tol': 1e-9},
    'physics': {
        'elasticity': True,
        'plasticity': True,
        'thermal': True
    }
}

sim = GeodynamicsSimulation(
    grid=grid,
    materials=materials,
    markers=swarm,
    BC=BC,
    config=config
)

# Run simulation
output_dir = 'results/'
sim.run(output_dir, checkpoint_interval=20)

# Or step through manually
for i in range(100):
    sim.step()
    if i % 10 == 0:
        state = sim.get_state()
        print(f"Step {i}: V_max = {state['vmax']:.3e}")
        sim.save_state(f"checkpoint_{i:04d}.h5")
```
```

---

## Part 7: Best Practices for Speckit Prompts

### 7.1 Be Specific
- ❌ "Create a solver"
- ✅ "Create StokesMatrixAssembler class that discretizes 2D Stokes momentum
  equations on a staggered grid using 4-point finite difference stencils,
  returning a scipy.sparse.csr_matrix for a (3·Nx·Ny) system"

### 7.2 Include Numbers & Acceptance Criteria
- ❌ "The code should be fast"
- ✅ "Assembly of 100×100 grid: < 1 second; 10 Picard iterations: < 5 seconds"

### 7.3 Reference the Original
- Cite MATLAB functions by name
- Reference equation numbers or papers if applicable
- Mention input/output formats

### 7.4 Provide Examples
- Show intended usage in code snippets
- Include test case results
- Illustrate API design

### 7.5 Separate Concerns
- One Speckit prompt = one class/module
- Dependencies clearly stated
- Avoid circular dependencies

### 7.6 Define Constraints
- Data types (NumPy arrays, specific dtypes)
- Conventions (SI units, indexing order)
- External libraries (scipy, numba allowed/forbidden)
- Performance budgets

---

## Part 8: Prompt Checklist

Before submitting a prompt to Speckit, verify:

- [ ] **Context**: 1-2 paragraphs explaining what/why
- [ ] **Specification**: Clear statement of what to build
- [ ] **Requirements**: Bulleted, functional and non-functional
- [ ] **Constraints**: Technical, design, performance limits
- [ ] **Acceptance Criteria**: 3-5 verifiable tests
- [ ] **Example Usage**: Complete code snippet showing API
- [ ] **Scope**: Achievable in one coding session (not 10+ files)
- [ ] **Dependencies**: Clear what other modules this depends on
- [ ] **Testing**: How to verify correctness

---

## Part 9: Recommended Speckit Prompt Order

1. **StokesGrid** (foundational)
2. **Material & Phase** (data types)
3. **Marker & MarkerSwarm** (data types)
4. **Interpolation Utilities** (used by both solver and markers)
5. **StokesMatrixAssembler** (core numeric)
6. **StokesNonlinearSolver** (core numeric, uses assembler)
7. **Rheology Models** (used by solver)
8. **Time Stepper** (orchestrates, uses all above)
9. **I/O & Visualization** (saves, plots outputs)
10. **Optimization** (profile, then rewrite hot paths with Numba/GPU)

---

## Part 10: Questions for Refining Prompts

When designing a Speckit prompt, ask:

1. **Correctness**: How do I verify this matches MATLAB behavior?
   - Compare numerical outputs to test cases
   - Unit tests with analytical solutions
   
2. **Interface**: How will this be used in the next component?
   - Who calls this? What do they pass?
   - What do they expect to get back?
   
3. **Performance**: What are the bottlenecks?
   - Can this be done in NumPy/vectorized?
   - Is a loop required, or can we Numba-JIT it?
   
4. **Extensibility**: What might change in the future?
   - 3D extension? Need to parameterize 2D assumptions?
   - Different BC types? Use inheritance/strategy pattern?
   
5. **Testing**: What tests ship with this?
   - Unit tests for each public method
   - Integration test with adjacent components
   - Regression test vs. MATLAB outputs

---

## Summary: Ready to Start

You now have:

1. **SiSteR Knowledge**: Physics, numerics, algorithms
2. **Speckit Strategy**: How to decompose into testable prompts
3. **Example Prompts**: Ready to iterate on for first 3 core modules

**Next**: Pick one prompt (e.g., StokesGrid), refine it with specifics from your
MATLAB code, and submit to Speckit. Then iterate based on feedback.

**Good luck!** 🚀
