# Phase 2: API Reference - Complete Module Documentation

## Table of Contents
- [Phase 2A: Linear Solver](#phase-2a-linear-solver)
- [Phase 2B: Time Stepping](#phase-2b-time-stepping)
- [Phase 2C: Rheology](#phase-2c-rheology)
- [Phase 2D: Thermal Solver](#phase-2d-thermal-solver)
- [Phase 2E: Performance](#phase-2e-performance)
- [Phase 2F: Validation](#phase-2f-validation)

---

# Phase 2A: Linear Solver

**Module:** `sister_py.linear_solver`

## Class: SolverSystem

Main interface for solving sparse linear systems with automatic solver selection.

### Constructor
```python
SolverSystem(
    matrix: sparse.csr_matrix,
    verbose: bool = False
)
```

**Parameters:**
- `matrix` (sparse.csr_matrix): System matrix A in Ax=b
- `verbose` (bool): Enable diagnostic output

**Example:**
```python
from scipy import sparse
import numpy as np
from sister_py import SolverSystem

# Create 5×5 test problem
A = sparse.csr_matrix(np.array([
    [4, -1,  0, -1,  0],
    [-1, 4, -1,  0, -1],
    [0, -1,  4, -1,  0],
    [-1, 0, -1,  4, -1],
    [0, -1,  0, -1,  4]
]))
b = np.ones(5)

solver = SolverSystem(A, verbose=True)
```

### Methods

#### solve(b, tol=1e-6)
Solve Ax=b using auto-selected solver.

**Parameters:**
- `b` (ndarray): Right-hand side vector
- `tol` (float): Convergence tolerance

**Returns:**
- `solution` (ndarray): Solution vector x
- `info` (dict): Solver information {method, iterations, residual}

**Example:**
```python
x, info = solver.solve(b, tol=1e-6)
print(f"Solved with {info['method']} ({info['iterations']} iterations)")
```

#### solve_direct(b)
Solve using LU factorization (exact method).

**Parameters:**
- `b` (ndarray): Right-hand side

**Returns:**
- `x` (ndarray): Exact solution

**Use When:**
- Problem size < 5,000 DOFs
- Need exact solution
- Matrix is well-conditioned

**Example:**
```python
x_exact = solver.solve_direct(b)
print(f"Direct solve residual: {np.linalg.norm(A @ x_exact - b):.2e}")
```

#### solve_iterative_gmres(b, maxiter=None, tol=1e-6, restart=30)
GMRES iterative solver with preconditioning.

**Parameters:**
- `b` (ndarray): Right-hand side
- `maxiter` (int): Maximum iterations (default: 2*n)
- `tol` (float): Relative tolerance rtol
- `restart` (int): GMRES restart parameter

**Returns:**
- `x` (ndarray): Approximate solution
- `n_iter` (int): Number of iterations
- `residual` (float): Final residual norm

**Use When:**
- Non-symmetric matrices
- Ill-conditioned problems
- Preconditioning available

**Example:**
```python
x, n_iter, res = solver.solve_iterative_gmres(b, maxiter=100)
print(f"GMRES: {n_iter} iterations, residual={res:.2e}")
```

**Parameters:**
- `callback_type` (str): 'pr_norm' (default) or 'legacy'

#### solve_iterative_bicgstab(b, maxiter=None, tol=1e-6)
BiCG-STAB iterative solver.

**Parameters:**
- `b` (ndarray): Right-hand side
- `maxiter` (int): Maximum iterations
- `tol` (float): Relative tolerance

**Returns:**
- `x` (ndarray): Solution
- `n_iter` (int): Iterations
- `residual` (float): Residual norm

**Use When:**
- Low memory available
- Need quasi-optimal residual
- Moderate-size problems

**Example:**
```python
x, n_iter, res = solver.solve_iterative_bicgstab(b)
print(f"BiCG-STAB converged in {n_iter} iterations")
```

#### solve_multigrid(b, n_levels=3, verbose=False)
Multigrid V-cycle solver.

**Parameters:**
- `b` (ndarray): Right-hand side
- `n_levels` (int): Number of multigrid levels
- `verbose` (bool): Print V-cycle info

**Returns:**
- `x` (ndarray): Solution
- `residuals` (list): Residual history

**Use When:**
- Poisson-like equations
- Need O(n) complexity
- Want lowest execution time

**Example:**
```python
x, residuals = solver.solve_multigrid(b, n_levels=4)
print(f"Multigrid convergence rate: {residuals[-1]/residuals[0]:.2e}")
```

---

## Class: LinearSystemAssembler

Utility for assembling Stokes system matrix.

### Constructor
```python
LinearSystemAssembler(
    nx: int, ny: int,
    dx: float = 1.0, dy: float = 1.0
)
```

### Methods

#### assemble_stokes_operator(viscosity_field)
Build Stokes operator with velocity-pressure coupling.

**Parameters:**
- `viscosity_field` (ndarray): Viscosity on grid [ny, nx]

**Returns:**
- `matrix` (sparse.csr_matrix): Stokes operator
- `rhs` (ndarray): Right-hand side vector

**Example:**
```python
assembler = LinearSystemAssembler(nx=50, ny=50, dx=2e3, dy=2e3)
eta = np.ones((50, 50)) * 1e21
A, b = assembler.assemble_stokes_operator(eta)
print(f"Matrix shape: {A.shape}, nnz: {A.nnz}")
```

---

# Phase 2B: Time Stepping

**Module:** `sister_py.time_stepper`

## Class: TimeIntegrator

Main coordinator for time stepping with multiple schemes.

### Constructor
```python
TimeIntegrator(
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    viscosity_field: np.ndarray,
    adaptive_timestepping: bool = False,
    verbose: bool = False
)
```

**Parameters:**
- `grid_x`, `grid_y` (ndarray): Grid coordinates [m]
- `viscosity_field` (ndarray): Initial viscosity [ny, nx]
- `adaptive_timestepping` (bool): Enable adaptive dt
- `verbose` (bool): Print diagnostics

**Example:**
```python
x = np.linspace(0, 100e3, 50)
y = np.linspace(0, 100e3, 50)
eta = np.ones((50, 50)) * 1e21

integrator = TimeIntegrator(x, y, eta, verbose=True)
```

### Methods

#### step(velocity_x, velocity_y, dt, scheme='backward_euler')
Advance one time step.

**Parameters:**
- `velocity_x`, `velocity_y` (ndarray): Velocity field [ny, nx]
- `dt` (float): Time step [s]
- `scheme` (str): 'forward_euler' or 'backward_euler'

**Returns:**
- `result` (TimeStepResult): Result object with:
  - `velocity_x`, `velocity_y`: Updated velocity
  - `stress`: Deviatoric stress tensor
  - `pressure`: Pressure field
  - `strain_rate`: Strain rate second invariant
  - `dt_used`: Actual dt used (if adaptive)

**Example:**
```python
vx, vy = np.zeros((50,50)), np.ones((50,50))*1e-9
result = integrator.step(vx, vy, dt=1e12)
print(f"Max stress: {np.max(result.stress):.2e} Pa")
```

#### solve_stokes(velocity_x, velocity_y)
Solve Stokes equations for given velocity field.

**Returns:**
- `pressure` (ndarray): Pressure field [ny, nx]
- `stress` (ndarray): Deviatoric stress invariant

#### update_stress(velocity_field)
Update stress tensor from velocity.

**Parameters:**
- `velocity_field` (tuple): (vx, vy) arrays

**Returns:**
- `stress` (ndarray): Stress tensor invariant

#### advect_markers(markers, velocity, dt)
Advance marker positions.

**Parameters:**
- `markers` (ndarray): Marker positions [n_markers, 2]
- `velocity` (tuple): (vx_grid, vy_grid)
- `dt` (float): Time step [s]

**Returns:**
- `markers_new` (ndarray): Updated positions

---

## Class: AdvectionScheme

Handles marker-grid interpolation and advection.

### Constructor
```python
AdvectionScheme(
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    interpolation_method: str = 'bilinear'
)
```

### Methods

#### interpolate_velocity_to_markers(velocity_x, velocity_y, marker_positions)
Map grid velocity to marker locations.

**Returns:**
- `vx_markers`, `vy_markers` (ndarray): Velocity at markers [n_markers]

**Example:**
```python
advection = AdvectionScheme(x, y)
vx_m, vy_m = advection.interpolate_velocity_to_markers(vx_grid, vy_grid, marker_pos)
```

#### advect_marker_positions(marker_positions, velocity, dt)
Move markers following velocity field.

**Returns:**
- `positions_new` (ndarray): Updated marker positions [n_markers, 2]

#### interpolate_strain_rate_to_markers(strain_rate_grid, marker_positions)
Map grid strain rate to markers.

---

## Class: MarkerReseeding

Maintain uniform marker distribution.

### Methods

#### detect_empty_cells(markers, grid_shape)
Find cells with no markers.

#### reseed_empty_cells(empty_cells, existing_markers)
Create new markers in depleted regions.

#### remove_duplicates(markers, min_distance)
Remove closely-spaced duplicate markers.

---

# Phase 2C: Rheology

**Module:** `sister_py.rheology`

## Class: RheologyModel

Main rheology coordinator.

### Constructor
```python
RheologyModel(
    material_props: Optional[ThermalMaterialProperties] = None,
    phase_field: Optional[np.ndarray] = None
)
```

### Methods

#### compute_viscosity(temperature, pressure=0, strain_rate=1e-15)
Temperature and strain-rate dependent viscosity.

**Parameters:**
- `temperature` (ndarray or float): Temperature [K]
- `pressure` (ndarray or float): Pressure [Pa]
- `strain_rate` (ndarray or float): Strain rate second invariant [s⁻¹]

**Returns:**
- `viscosity` (ndarray or float): Effective viscosity [Pa·s]

**Example:**
```python
rheology = RheologyModel()
T = np.linspace(300, 1600, 100)
eta = rheology.compute_viscosity(T)
print(f"Viscosity range: {eta.min():.1e} - {eta.max():.1e} Pa·s")
```

#### apply_yield_criterion(stress_ii, pressure)
Apply plasticity cutoff.

**Parameters:**
- `stress_ii` (ndarray): Stress second invariant [Pa]
- `pressure` (ndarray): Pressure [Pa]

**Returns:**
- `stress_ii_capped` (ndarray): Capped stress [Pa]

#### update_stress(strain_rate, stress_old, dt, temperature)
Update deviatoric stress (Maxwell model).

**Parameters:**
- `strain_rate` (ndarray): Strain rate tensor
- `stress_old` (ndarray): Previous stress
- `dt` (float): Time step [s]
- `temperature` (ndarray): Temperature [K]

**Returns:**
- `stress_new` (ndarray): Updated stress

#### compute_strain_rate(velocity_gradient)
Compute strain rate invariant from velocity gradient.

**Parameters:**
- `velocity_gradient` (ndarray): ∇u tensor

**Returns:**
- `strain_rate_ii` (ndarray): Strain rate second invariant [s⁻¹]

---

## Class: TemperatureDependentViscosity

Arrhenius law for viscosity.

### Constructor
```python
TemperatureDependentViscosity(
    eta_0: float = 1e21,
    T_ref: float = 1273.15,
    E_a: float = 500e3,
    R: float = 8.314,
    n: float = 1.0,
    verbose: bool = False
)
```

**Parameters:**
- `eta_0` (float): Reference viscosity [Pa·s]
- `T_ref` (float): Reference temperature [K]
- `E_a` (float): Activation energy [J/mol]
- `R` (float): Gas constant 8.314 [J/mol/K]
- `n` (float): Power-law exponent (1=Newtonian, 3=typical dislocation)
- `verbose` (bool): Print debug info

### Methods

#### viscosity(temperature, strain_rate=1e-15)
Compute viscosity at given T.

**Parameters:**
- `temperature` (ndarray or float): Temperature [K]
- `strain_rate` (ndarray or float): Strain rate [s⁻¹]

**Returns:**
- `eta` (ndarray or float): Viscosity [Pa·s]

**Formula:**
```
η(T, ė) = η₀ * (ė/ė_ref)^(1/n-1) * exp(E_a/(n*R*T))
```

#### d_viscosity_dT(temperature)
Temperature derivative (for Jacobian).

**Returns:**
- `d_eta_dT` (ndarray): Derivative [Pa·s/K]

---

## Class: YieldCriterion

Drucker-Prager yield criterion.

### Constructor
```python
YieldCriterion(
    cohesion: float = 10e6,
    friction_coeff: float = 0.6
)
```

**Parameters:**
- `cohesion` (float): Cohesion [Pa]
- `friction_coeff` (float): Friction coefficient

### Methods

#### yield_stress(pressure)
Compute yield stress at given pressure.

**Parameters:**
- `pressure` (float): Confining pressure [Pa]

**Returns:**
- `tau_yield` (float): Yield stress [Pa]

**Formula:**
```
τ_yield = C + μ * P
```

#### effective_viscosity(pressure, strain_rate_ii, background_viscosity)
Cap viscosity at yield criterion.

**Returns:**
- `eta_eff` (float): Effective viscosity [Pa·s]

---

## Class: ElasticStressAccumulation

Maxwell viscoelastic model.

### Constructor
```python
ElasticStressAccumulation(
    shear_modulus: float = 30e9,
    viscosity: float = 1e21
)
```

**Parameters:**
- `shear_modulus` (float): Shear modulus G [Pa]
- `viscosity` (float): Viscosity η [Pa·s]

### Methods

#### update(strain_rate, stress_old, dt)
Update stress (Maxwell model).

**Parameters:**
- `strain_rate` (ndarray): Strain rate [s⁻¹]
- `stress_old` (ndarray): Previous stress [Pa]
- `dt` (float): Time step [s]

**Returns:**
- `stress_new` (ndarray): Updated stress [Pa]

**Formula:**
```
σ̇_D = 2G(ė_D - σ_D/(2η))
```

#### relaxation_time()
Compute Maxwell relaxation time.

**Returns:**
- `tau_relax` (float): Relaxation time τ = η/G [s]

---

# Phase 2D: Thermal Solver

**Module:** `sister_py.thermal_solver`

## Class: ThermalModel

Main thermal system coordinator.

### Constructor
```python
ThermalModel(
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    initial_temperature: np.ndarray,
    material_props: Optional[ThermalMaterialProperties] = None,
    verbose: bool = False
)
```

**Parameters:**
- `grid_x`, `grid_y` (ndarray): Grid coordinates [m]
- `initial_temperature` (ndarray): T field [K]
- `material_props` (ThermalMaterialProperties): Material database
- `verbose` (bool): Diagnostics

**Example:**
```python
x = np.linspace(0, 100e3, 50)
y = np.linspace(0, 100e3, 50)
T_init = 1500 - 10*y[:, None]

thermal = ThermalModel(x, y, T_init)
```

### Methods

#### solve_step(phase_field, velocity_x=None, velocity_y=None, heat_source=None, dt=1e12, use_advection=False)
Solve thermal equation for one time step.

**Parameters:**
- `phase_field` (ndarray): Material phases [ny, nx]
- `velocity_x`, `velocity_y` (ndarray): Velocity field [m/s]
- `heat_source` (ndarray): Internal heating [W/m³]
- `dt` (float): Time step [s]
- `use_advection` (bool): Include advection term

**Returns:**
- `result` (ThermalFieldData): Result with:
  - `temperature`: Temperature field [K]
  - `heat_flux_x`, `heat_flux_y`: Heat flux [W/m²]
  - `heat_generation`: Heating field [W/m³]
  - `time`: Simulation time [s]

**Example:**
```python
result = thermal.solve_step(
    phase_field=phase,
    velocity_x=vx, velocity_y=vy,
    dt=1e12,
    use_advection=True
)
T_new = result.temperature
```

#### set_boundary_conditions(bc_list)
Set thermal boundary conditions.

**Parameters:**
- `bc_list` (list): List of ThermalBoundaryCondition objects

#### apply_dirichlet_bc(T, bc)
Apply Dirichlet BC to temperature field.

---

## Class: HeatDiffusionSolver

Pure diffusion (no advection).

### Methods

#### assemble_laplace_operator(k_field, grid_x, grid_y)
Build Laplacian system.

**Returns:**
- `matrix` (sparse.csr_matrix): Discretized Laplacian
- `rhs` (ndarray): Right-hand side

#### solve_steady_state(k_field, grid_x, grid_y, heat_source)
Solve ∇·(k∇T) = -Q.

**Returns:**
- `T` (ndarray): Temperature field [K]

#### solve_transient(T_old, k_field, cp_field, rho_field, grid_x, grid_y, heat_source, dt)
Solve time-dependent heat equation (backward Euler).

**Formula:**
```
(ρ*cp/dt + A) * T^{n+1} = (ρ*cp/dt) * T^n + Q
```

**Returns:**
- `T_new` (ndarray): Temperature at new time step [K]

---

## Class: AdvectionDiffusionSolver

Coupled advection-diffusion with SUPG stabilization.

### Methods

#### assemble_advection_diffusion(velocity_x, velocity_y, k_field, grid_x, grid_y)
Build advection-diffusion operator.

**Returns:**
- `matrix` (sparse.csr_matrix): System matrix
- `rhs` (ndarray): RHS

#### solve_advection_diffusion(T_old, velocity_x, velocity_y, k_field, grid_x, grid_y, heat_source, dt)
Solve coupled system with SUPG.

**Returns:**
- `T_new` (ndarray): Temperature [K]

---

## Class: ThermalProperties

Material thermal properties (dataclass).

### Attributes
```python
@dataclass
class ThermalProperties:
    k: float = 3.0              # Thermal conductivity [W/m/K]
    k_aniso_ratio: float = 1.0  # Anisotropy ratio
    cp: float = 1000.0          # Heat capacity [J/kg/K]
    rho: float = 2800.0         # Density [kg/m³]
    alpha: float = 3e-5         # Thermal expansion [1/K]
    T_ref: float = 273.15       # Reference temperature [K]
```

---

## Class: ThermalBoundaryCondition

Thermal BC specification.

### Constructor
```python
ThermalBoundaryCondition(
    boundary: str,
    bc_type: str,
    value: float = 0.0,
    ambient_temp: float = 273.15,
    h_coeff: float = 0.0
)
```

**Parameters:**
- `boundary` (str): 'top', 'bottom', 'left', 'right'
- `bc_type` (str): 'dirichlet', 'neumann', or 'robin'
- `value` (float): Temperature [K] for Dirichlet, flux [W/m²] for Neumann
- `ambient_temp` (float): Ambient for Robin [K]
- `h_coeff` (float): Convection coefficient for Robin [W/m²/K]

**Example:**
```python
bc_surface = ThermalBoundaryCondition(
    boundary='top',
    bc_type='dirichlet',
    value=273.15  # 0°C
)

bc_bottom = ThermalBoundaryCondition(
    boundary='bottom',
    bc_type='neumann',
    value=0.06  # 60 mW/m² heat flux
)
```

---

## Functions

### compute_thermal_conductivity(temperature, mineral='olivine', pressure=0)
Temperature-dependent conductivity.

**Parameters:**
- `temperature` (float): Temperature [K]
- `mineral` (str): 'olivine', 'basalt', 'granite'
- `pressure` (float): Pressure [Pa]

**Returns:**
- `k` (float): Conductivity [W/m/K]

### compute_heat_capacity(temperature, mineral='olivine')
Temperature-dependent heat capacity.

**Returns:**
- `cp` (float): Heat capacity [J/kg/K]

### estimate_thermal_time_scale(L, k=3.0, rho=2800, cp=1000)
Estimate thermal diffusion time scale τ ~ L²/α.

**Parameters:**
- `L` (float): Length scale [m]
- Others: Material properties

**Returns:**
- `tau` (float): Time scale [s]

### interpolate_temperature_to_markers(T_grid, marker_positions, grid_x, grid_y)
Bilinear interpolation of temperature.

**Returns:**
- `T_markers` (ndarray): Temperature at marker locations [K]

---

# Phase 2E: Performance

**Module:** `sister_py.performance`

## Class: PerformanceProfiler

Context manager for timing code blocks.

### Usage
```python
from sister_py import PerformanceProfiler

profiler = PerformanceProfiler()

with profiler.timer("stokes_solve"):
    # Solve Stokes system
    x = solver.solve(b)

with profiler.timer("thermal_solve"):
    # Solve thermal equation
    T = thermal.solve_step(...)

print(profiler.get_summary())
```

### Methods

#### timer(label)
Context manager for timing a code block.

#### reset()
Clear timing history.

#### get_summary()
Print timing report.

**Output:**
```
Performance Summary:
  stokes_solve:      1.234 s (calls: 100)
  thermal_solve:     0.567 s (calls: 100)
```

---

## Class: PerformanceMetrics

Efficiency calculation utilities.

### Methods

#### compute_l2_norm(solution, reference)
Compute L2 error norm.

**Parameters:**
- `solution` (ndarray): Computed solution
- `reference` (ndarray): Exact solution

**Returns:**
- `error` (float): ||solution - reference||₂

#### compute_gflops(n_flops, time_seconds)
Calculate GFLOPS (billion FLOPs per second).

**Returns:**
- `gflops` (float): Throughput [GFLOPS]

#### estimate_throughput(matrix_size, time_seconds)
Estimate GB/s bandwidth.

---

## Class: MultiGridPreconditioner

Multigrid V-cycle solver.

### Constructor
```python
MultiGridPreconditioner(
    matrix: sparse.csr_matrix,
    n_levels: int = 3
)
```

### Methods

#### setup()
Build restriction/prolongation operators.

#### apply_vcycle(b, initial_guess=None, verbose=False)
Execute one V-cycle.

**Parameters:**
- `b` (ndarray): Right-hand side
- `initial_guess` (ndarray): Starting solution
- `verbose` (bool): Print V-cycle info

**Returns:**
- `x` (ndarray): Solution approximation

---

## Class: OptimizedSolver

Automatic solver selection and execution.

### Constructor
```python
OptimizedSolver(
    matrix: sparse.csr_matrix,
    method: str = 'auto'
)
```

**Parameters:**
- `matrix` (sparse.csr_matrix): System matrix
- `method` (str): 'auto', 'direct', 'gmres', 'bicgstab', 'multigrid'

### Methods

#### solve(b, tol=1e-6)
Solve Ax=b.

**Returns:**
- `x` (ndarray): Solution
- `info` (dict): Solver info {method, iterations, residual}

#### solve_direct(b)
LU factorization solve.

#### solve_iterative_gmres(b, maxiter=None, tol=1e-6)
GMRES solve.

#### solve_iterative_bicgstab(b, maxiter=None, tol=1e-6)
BiCG-STAB solve.

#### solve_multigrid(b, n_levels=3)
Multigrid solve.

---

## Functions

### benchmark_solver(A, b)
Compare all solver methods on same problem.

**Returns:**
- `results` (dict): Timing and iteration counts for each method

**Example:**
```python
from sister_py import benchmark_solver

results = benchmark_solver(A_stokes, b_rhs)
for method, data in results.items():
    print(f"{method:12s}: {data['time']:.4f} s")
```

### estimate_memory_usage(n_dof, n_fields=3)
Estimate peak memory for problem.

**Returns:**
- `memory_mb` (float): Memory requirement [MB]

### estimate_flops(matrix, iterations=1)
Estimate FLOP count for solve.

**Returns:**
- `flops` (float): Operation count

### profile_code(label=None)
Decorator for automatic function profiling.

**Usage:**
```python
from sister_py import profile_code

@profile_code("thermal_step")
def my_thermal_solve(...):
    return thermal.solve_step(...)

# Automatically timed and profiled
```

---

# Phase 2F: Validation

**Module:** `sister_py.validation`

## Class: AnalyticalSolution (ABC)

Abstract base class for benchmark solutions.

### Abstract Methods

```python
@abstractmethod
def evaluate(self, x: np.ndarray, y: np.ndarray, t: float = 0.0) -> np.ndarray:
    """Evaluate solution at (x,y,t)."""
    pass

@abstractmethod
def evaluate_x_derivative(self, x: np.ndarray, y: np.ndarray, t: float = 0.0) -> np.ndarray:
    """Evaluate ∂u/∂x."""
    pass

@abstractmethod
def evaluate_y_derivative(self, x: np.ndarray, y: np.ndarray, t: float = 0.0) -> np.ndarray:
    """Evaluate ∂u/∂y."""
    pass
```

---

## Class: PoiseuilleFlow

Parabolic channel flow benchmark.

### Constructor
```python
PoiseuilleFlow(
    U_max: float = 1.0,
    h: float = 1.0
)
```

**Parameters:**
- `U_max` (float): Maximum velocity
- `h` (float): Channel half-height

### Methods

#### evaluate(x, y, t=0)
Velocity profile: u(y) = U_max * (1 - (y/h)²)

**Returns:**
- `u` (ndarray): Velocity field

#### evaluate_x_derivative(x, y, t=0)
∂u/∂x = 0 (fully developed flow)

#### evaluate_y_derivative(x, y, t=0)
∂u/∂y = -2*U_max*y/h²

**Example:**
```python
from sister_py import PoiseuilleFlow

poiseuille = PoiseuilleFlow(U_max=1.0, h=0.1)
y = np.linspace(-0.1, 0.1, 100)
u = poiseuille.evaluate(np.zeros_like(y), y)
```

---

## Class: ThermalDiffusion

Heat equation analytical solution.

### Constructor
```python
ThermalDiffusion(
    T0: float = 0.0,
    T1: float = 1.0,
    alpha: float = 1e-6
)
```

**Parameters:**
- `T0` (float): Initial temperature
- `T1` (float): Boundary temperature
- `alpha` (float): Thermal diffusivity [m²/s]

### Methods

#### evaluate(x, y, t=0)
Solution: T(x,t) = T0 + (T1-T0)*erfc(x/√(4αt))

**Example:**
```python
thermal_bench = ThermalDiffusion(T0=0, T1=1.0, alpha=1e-6)
x = np.linspace(0, 0.1, 100)
T = thermal_bench.evaluate(x, np.zeros_like(x), t=100)
```

---

## Class: CavityFlow

Lid-driven cavity benchmark.

### Constructor
```python
CavityFlow(
    Re: float = 1.0
)
```

**Parameters:**
- `Re` (float): Reynolds number

### Methods

#### evaluate(x, y, t=0)
Stream function ψ(x,y) approximation.

**Returns:**
- `psi` (ndarray): Stream function

---

## Class: ErrorMetrics

Error norm computation.

### Static Methods

#### compute_l2_norm(computed, exact)
L2 error: ||computed - exact||₂ = √(Σ(error²))

**Parameters:**
- `computed` (ndarray): Numerical solution
- `exact` (ndarray): Analytical solution

**Returns:**
- `error_l2` (float): L2 norm

#### compute_linf_norm(computed, exact)
Linf error: ||computed - exact||∞ = max|error|

**Returns:**
- `error_linf` (float): Maximum error

#### compute_relative_error(computed, exact)
Relative error: ||error|| / ||exact||

**Returns:**
- `rel_error` (float): Normalized error

**Example:**
```python
from sister_py import ErrorMetrics

computed = np.random.rand(100) * 2  # [0, 2]
exact = np.sin(np.linspace(0, np.pi, 100))

L2 = ErrorMetrics.compute_l2_norm(computed, exact)
Linf = ErrorMetrics.compute_linf_norm(computed, exact)
rel_L2 = ErrorMetrics.compute_relative_error(computed, exact)

print(f"L2={L2:.4f}, Linf={Linf:.4f}, rel_L2={rel_L2:.4f}")
```

---

## Class: ConvergenceStudy

Grid refinement analysis.

### Constructor
```python
ConvergenceStudy(
    analytical_solution: AnalyticalSolution
)
```

### Methods

#### add_convergence_data(grid_spacing, L2_error, Linf_error)
Store error at one mesh size.

**Parameters:**
- `grid_spacing` (float): dx [m]
- `L2_error` (float): L2 error norm
- `Linf_error` (float): Linf error norm

#### estimate_convergence_rates()
Compute rates: r = log(e₂/e₁)/log(dx₂/dx₁)

**Returns:**
- `rates` (dict): Convergence rates for each norm

**Example:**
```python
from sister_py import ConvergenceStudy, PoiseuilleFlow

study = ConvergenceStudy(PoiseuilleFlow())

for n in [10, 20, 40, 80]:
    dx = 1.0 / n
    x = np.linspace(0, 1, n)
    y_exact = poiseuille.evaluate(np.zeros(n), x)
    y_computed = compute_solution(n)  # Your solver
    
    L2 = ErrorMetrics.compute_l2_norm(y_computed, y_exact)
    study.add_convergence_data(dx, L2, 0)

rates = study.estimate_convergence_rates()
print(f"Convergence rate: {rates['L2']}")  # Should be ~2.0
```

---

## Class: ValidationReport

Comprehensive accuracy report.

### Constructor
```python
ValidationReport(
    test_name: str,
    error_metrics: ErrorMetrics,
    convergence_rate: Optional[float] = None
)
```

### Methods

#### generate_report()
Create detailed report.

**Returns:**
- `report` (str): Full report text

**Output Example:**
```
═══════════════════════════════════════
Validation Report: Poiseuille Flow
═══════════════════════════════════════
Test Case:       Channel Flow, h=0.1 m, U_max=1 m/s
Mesh:            100×100 grid, dx=0.001 m
Errors:
  L2:            3.4e-5 (EXCELLENT)
  Linf:          5.2e-5 (EXCELLENT)
  Relative L2:   1.2e-3
Convergence:     2.04 (VERIFIED 2nd order)
Status:          ✓ PASS
═══════════════════════════════════════
```

---

## Class: BenchmarkTestCase

Execute complete benchmark suite.

### Methods

#### test_poiseuille(nx=50, ny=50)
Validate Poiseuille flow channel.

**Returns:**
- `report` (str): Validation results

#### test_thermal_diffusion(n_grids=4)
Convergence study on thermal diffusion.

**Returns:**
- `convergence_rates` (dict): Rate for each norm

#### test_cavity_flow(Re_list=[1, 10, 100])
Multi-Reynolds cavity flow validation.

**Returns:**
- `results` (dict): Cavity statistics per Re

---

## Functions

### run_full_validation_suite(nx=50, ny=50)
Execute all validation benchmarks.

**Returns:**
- `results` (dict): Complete test results

**Example:**
```python
from sister_py import run_full_validation_suite

results = run_full_validation_suite(nx=100, ny=100)

for test_name, status in results.items():
    symbol = "✓" if status else "✗"
    print(f"{symbol} {test_name}")
```

### generate_validation_report(test_name, numerical_solution, analytical_solution)
Generate single test report.

**Returns:**
- `report` (str): Formatted report text

---

## Constants

```python
# Accuracy thresholds
ACCURACY_EXCELLENT = 1e-4    # L2 error < 1e-4
ACCURACY_GOOD = 1e-2         # L2 error < 1e-2
ACCURACY_ACCEPTABLE = 1e-1   # L2 error < 1e-1

# Expected convergence rates
CONVERGENCE_LINEAR = 1.0     # 1st order
CONVERGENCE_QUADRATIC = 2.0  # 2nd order
CONVERGENCE_CUBIC = 3.0      # 3rd order
```

---

## Quick Reference Examples

```python
# Complete validation workflow
from sister_py import *

# 1. Create analytical solution
poiseuille = PoiseuilleFlow(U_max=1.0, h=0.1)

# 2. Run convergence study
study = ConvergenceStudy(poiseuille)
for n in [20, 40, 80, 160]:
    # ... solve on grid with n points
    study.add_convergence_data(1/n, L2_error, Linf_error)

# 3. Analyze convergence
rates = study.estimate_convergence_rates()

# 4. Generate report
report = ValidationReport("Poiseuille", errors, rates['L2'])
print(report.generate_report())

# 5. Full suite
results = run_full_validation_suite(nx=100, ny=100)
```

