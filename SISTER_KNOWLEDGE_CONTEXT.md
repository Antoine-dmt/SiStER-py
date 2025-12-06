# SiSteR Code Knowledge Base & Stokes Equation Concepts

## Part 1: SiSteR Overview

### What is SiSteR?

**SiSteR** stands for "**S**imple Stokes solver with **E**xotic **R**heologies" and is a MATLAB-based geodynamic code designed for long-term lithosphere and mantle deformation simulations.

**Authors**: J.-A. Olive, B.Z. Klein, E. Mittelstaedt, M. Behn, G. Ito, S. Howell

**Purpose**: Simulates geological processes over millions of years (Myr), including:
- Continental rifting
- Lithospheric delamination
- Subduction zone dynamics
- Mantle convection with complex rheologies

### Key Characteristics

1. **Eulerian Grid-Based**: Uses a fixed Cartesian grid to solve momentum and continuity equations
2. **Lagrangian Markers**: Tracks material properties and histories through discrete particles
3. **Non-linear Rheology**: Supports:
   - **Ductile creep** (diffusion + dislocation creep via power-law flow laws)
   - **Elasticity** (recoverable strain with shear modulus)
   - **Plasticity** (Mohr-Coulomb yielding for brittle failure)
   - **Visco-elasto-plastic** (VEP) coupling
4. **Staggered Grid**: Velocity components and pressure on different grid points (reduces oscillations)
5. **Finite Difference**: Uses 2D finite differences for spatial discretization

---

## Part 2: Stokes Equation Fundamentals

### The Stokes Equations

Stokes equations describe slow, viscous flow (low Reynolds number), appropriate for geological timescales:

$$\nabla \cdot \boldsymbol{\sigma} + \rho \mathbf{g} = 0 \quad \text{(momentum balance)}$$

$$\nabla \cdot \mathbf{v} = 0 \quad \text{(mass conservation/continuity)}$$

Where:
- $\boldsymbol{\sigma}$ = deviatoric stress tensor = $2\eta \mathbf{\dot{\varepsilon}}$
- $\eta$ = dynamic viscosity (rheology-dependent)
- $\mathbf{\dot{\varepsilon}}$ = strain rate tensor
- $\rho$ = density
- $\mathbf{g}$ = gravity
- $\mathbf{v}$ = velocity field

### In 2D Component Form

**Momentum X:**
$$\frac{\partial \sigma_{xx}}{\partial x} + \frac{\partial \sigma_{xy}}{\partial y} + \rho g_x = 0$$

**Momentum Y:**
$$\frac{\partial \sigma_{xy}}{\partial x} + \frac{\partial \sigma_{yy}}{\partial y} + \rho g_y = 0$$

**Continuity:**
$$\frac{\partial v_x}{\partial x} + \frac{\partial v_y}{\partial y} = 0$$

### Strain Rate Tensor

$$\mathbf{\dot{\varepsilon}} = \begin{pmatrix} 
\dot{\varepsilon}_{xx} & \dot{\varepsilon}_{xy} \\
\dot{\varepsilon}_{xy} & \dot{\varepsilon}_{yy}
\end{pmatrix}$$

Where:
$$\dot{\varepsilon}_{xx} = \frac{\partial v_x}{\partial x}, \quad \dot{\varepsilon}_{yy} = \frac{\partial v_y}{\partial y}, \quad \dot{\varepsilon}_{xy} = \frac{1}{2}\left(\frac{\partial v_x}{\partial y} + \frac{\partial v_y}{\partial x}\right)$$

### Deviatoric Stress

$$\sigma_{ij} = 2\eta \dot{\varepsilon}_{ij} \quad \text{(deviatoric)}$$

$$p = -\frac{1}{3}\text{Tr}(\boldsymbol{\sigma}) \quad \text{(pressure, isotropic)}$$

---

## Part 3: SiSteR Computational Strategy

### 3.1 Staggered Grid Layout

SiSteR uses a **staggered (MAC) grid** to prevent pressure oscillations:

```
  j-1    j    j+1
i+1 +----+----+
    | u  |  v |
  i +----P----+  <- Pressure (normal nodes)
    | u  |  v |
i-1 +----+----+
```

- **Normal nodes (•)**: Pressure nodes at grid corners
- **Shear nodes (◊)**: Velocity component nodes at cell centers/edges
- $v_x$ and $v_y$ are staggered horizontally and vertically

### 3.2 Discretization via Finite Differences

The system is discretized on the grid:

**Example (Momentum in X on shear node):**
$$\frac{\sigma_{xx}(i,j+1/2) - \sigma_{xx}(i,j-1/2)}{\Delta x} + \frac{\sigma_{xy}(i+1/2,j) - \sigma_{xy}(i-1/2,j)}{\Delta y} = 0$$

This creates a **sparse linear system**: $\mathbf{L} \mathbf{S} = \mathbf{R}$

Where:
- $\mathbf{L}$ = Left-hand side matrix (FD discretization, depends on viscosity)
- $\mathbf{S}$ = Solution vector = $[p_1, v_{x,1}, v_{y,1}, p_2, v_{x,2}, v_{y,2}, ...]^T$
- $\mathbf{R}$ = Right-hand side (gravity, body forces)

### 3.3 Non-linear Iteration Strategy (Picard + Newton)

Since viscosity $\eta$ depends on stress/strain rate (non-linear), SiSteR uses:

**Step 1: Picard Iteration** (early iterations, robust)
```
Loop pit = 1 to Npicard_min:
    1. Compute viscosity η(ε̇, σ, T) from current strain rate/stress
    2. Assemble L and R matrices
    3. Solve: S_new = L^{-1} R
    4. Check convergence: ||L·S_new - R||_2 / ||R||_2 < tolerance
```

**Step 2: Approximate Newton** (switch at pit=Npicard_switch, faster convergence)
```
Loop pit = Npicard_min to Npicard_max:
    1. Update viscosity
    2. Assemble L and R
    3. Compute residual: Res = L·S - R
    4. Newton step: S_new = S - (L^{-1} Res)
    5. Check convergence
```

**Stopping Criteria**:
- L2 residual norm drops below `conv_crit_ResL2` (default 1e-9)
- AND minimum iterations exceeded
- OR maximum iterations reached (with warning)

### 3.4 Main Time Loop Structure

```matlab
for time_step = 1 to Nt
    
    1. Update material properties at nodes from markers
       [SiStER_material_props_on_nodes]
       
    2. SOLVE STOKES (non-linear)
       [SiStER_flow_solve]
       - Picard/Newton iterations
       - Compute v_x, v_y, p
       
    3. Interpolate strain rate to markers
       [interp_shear_nodes_to_markers: ε̇_II]
       
    4. Update marker stresses (elastic/plastic)
       [SiStER_update_marker_stresses]
       [SiStER_update_ep] if plasticity enabled
       
    5. OUTPUT (if requested)
       Save: v_x, v_y, p, T, stresses, viscosity, phase info
       
    6. Set adaptive time step
       [SiStER_set_timestep] based on CFL condition
       
    7. Rotate elastic stresses (if elasticity on)
       [SiStER_rotate_stresses]
       
    8. THERMAL DIFFUSION (if enabled)
       [SiStER_thermal_update]
       
    9. LAGRANGIAN ADVECTION
       [SiStER_move_remove_and_reseed_markers]
       - Move markers via interpolated velocities
       - Remove markers at boundaries
       - Reseed where marker density too low
       [SiStER_update_topography_markers]
       
end
```

### 3.5 Matrix Assembly: `SiStER_assemble_L_R`

This is the computational core:

**Inputs**:
- Grid parameters: `dx, dy, Nx, Ny`
- Viscosity at nodes: `etas` (shear), `etan` (normal)
- Density: `rho` (varies with material)
- Boundary conditions: `BC` (velocity/stress on edges)
- Stress history (if elastic): `srhs_xx, srhs_xy`

**Process**:
1. Pre-compute scaling coefficients:
   - $K_c = 2\eta_{min} / (dx_{max} + dy_{max})$ (momentum scaling)
   - $K_b = 4\eta_{min} / (dx_{max} + dy_{max})^2$ (continuity scaling)

2. Loop through each grid point:
   - **Continuity equation** for pressure (if interior point)
   - **X-momentum** for $v_x$ (if interior point)
   - **Y-momentum** for $v_y$ (if interior point)
   - **Boundary conditions** (if on edge)

3. Assemble sparse matrix `L` and vector `R` in COO format:
   - `Lii, Ljj, Lvv` = row indices, column indices, values
   - Convert to sparse matrix for efficient storage

4. Return:
   - `L`: sparse FD matrix (order 3N×3N)
   - `R`: RHS vector (3N×1)
   - `Kc, Kb`: scaling coefficients

---

## Part 4: Rheology Models in SiSteR

### 4.1 Ductile Creep (Non-Linear)

Power-law creep (dislocation + diffusion):

$$\dot{\varepsilon}_{II} = B \sigma_{II}^n \exp\left(-\frac{E}{nRT}\right)$$

Where:
- $B = A^{-1/n}$ (pre-exponential factor, material-dependent)
- $\sigma_{II}$ = second invariant of deviatoric stress
- $n$ = stress exponent (1 for diffusion, ~3.5 for dislocation)
- $E$ = activation energy
- $R$ = gas constant (8.314 J/mol·K)
- $T$ = temperature

**Effective viscosity**:
$$\eta_{eff} = \frac{\sigma_{II}}{2\dot{\varepsilon}_{II}}$$

### 4.2 Plasticity (Mohr-Coulomb)

Yield strength:
$$\sigma_Y = (C + \mu \cdot P) \cos(\arctan(\mu))$$

Where:
- $C$ = cohesion
- $\mu$ = friction coefficient
- $P$ = pressure

If stress exceeds yield: viscosity capped at:
$$\eta_{plas} = \frac{\sigma_Y}{2\dot{\varepsilon}_{II}}$$

### 4.3 Elasticity (VEP Coupling)

Elastic strain accumulates:
$$\sigma = \sigma^{elastic} + \sigma^{viscous} = 2G\varepsilon^{elastic} + 2\eta\dot{\varepsilon}^{viscous}$$

Stress rate in flowing material:
$$\dot{\sigma} = 2G\left(\dot{\varepsilon}_{total} - \dot{\varepsilon}^{plastic}\right) - \sigma \cdot \omega$$

Where $\omega$ is rotation rate. Stresses are rotated to follow material flow.

---

## Part 5: Marker-Based Tracking

### 5.1 Purpose

Markers track:
- Material phase (rock type)
- Composition/density
- Temperature (advected)
- Stress history (elastic strain, plastic strain)
- Plastic strain ($e_p$): cumulative inelastic deformation
- Strain rate history

### 5.2 Marker Operations

**Initialization** `[SiStER_initialize_marker_positions]`:
- Uniform distribution with `Mquad` markers per smallest grid quadrant
- Total: ~$Mquad^2 \times Nx \times Ny$ markers

**Interpolation - Markers → Nodes**:
- Bilinear interpolation of marker properties to grid nodes
- Weighted by volume fraction occupied
- Used to map phase-dependent properties (density, rheology)

**Interpolation - Nodes → Markers**:
- Interpolate computed fields (velocity, strain rate, stress) back to markers
- Uses solved nodal values to update marker physics

**Advection** (Lagrangian step):
```matlab
x_m^{t+Δt} = x_m^t + Δt · v_interp(x_m^t)
```

**Reseeding**:
- Markers removed if outside domain or spacing too irregular
- New markers added where density drops below `Mquad_crit`

---

## Part 6: Example: Continental Rifting

Default model configuration:

**Domain**: 170 km × 60 km
- Variable grid resolution (coarse far from rift, fine in center)
- Center refined to 500 m

**Materials**:
1. **Layer 1 (sticky air/water)**: 0-10 km
   - Low density (1000 kg/m³)
   - Very low viscosity (brittle-like via plasticity)
   
2. **Layer 2 (Lithosphere/Mantle)**: 10-60 km
   - High density (3300 kg/m³)
   - Strong dislocation creep ($n=3.5$)
   - Brittle plasticity above

**Initial Conditions**:
- Linear geothermal gradient: $T(y) = a_3 \cdot y^3$ with $a_3 = 1000/(30 km)^3$
- Weak fault zone at center, 60° dip, 1 km width

**Boundary Conditions**:
- Extension imposed on top (sticky layer)
- Fixed on sides and bottom

---

## Part 7: Key Computational Challenges

### 7.1 Non-Linearity
- Viscosity depends on stress/strain rate → requires iteration
- Picard iterations robust but slow
- Newton iterations faster but need good initial guess

### 7.2 Stiffness
- Elastic moduli ~10¹¹ Pa, viscosities ~10¹⁸-10²⁴ Pa·s
- Large parameter ranges require careful scaling (Kc, Kb)

### 7.3 Localization
- Plastic zones can narrow to few grid cells
- Requires fine resolution in weak zones
- Adaptive refinement needed

### 7.4 Advection Accuracy
- Marker advection can suffer from interpolation errors
- Time stepping limited by CFL: $\Delta t < 0.5 \cdot \Delta x / v_{max}$

### 7.5 Thermal Coupling
- Temperature affects viscosity exponentially (slow diffusion)
- Requires coupled solve (or operator splitting)

---

## Part 8: Python/OOP Redesign Opportunities

### Current MATLAB Limitations
1. **Functional/Script-Based**: No encapsulation, global variables
2. **Memory Inefficiency**: Dense matrices, no sparse optimization
3. **Portability**: MATLAB license required
4. **Performance**: Interpreted, limited parallelization

### Proposed OOP Structure (Python)

```python
class StokesGrid:
    """Manages staggered grid, spacing, indexing"""
    def __init__(self, xsize, ysize, grid_config)
    def assemble_system_matrix(self)
    def interpolate_to_markers(self)
    def interpolate_to_nodes(self)

class Rheology:
    """Base class for material behavior"""
    def viscosity(self, stress, strain_rate, temperature)
    def stress_update(self, strain_rate, dt)

class DuctileRheology(Rheology):
    """Power-law creep"""
    
class PlasticRheology(Rheology):
    """Mohr-Coulomb plasticity"""

class ElasticRheology(Rheology):
    """Elastic stress accumulation"""

class Material:
    """Material properties container"""
    def __init__(self, rho, rheology_params, phase)
    def get_viscosity(self, state)

class Marker:
    """Individual Lagrangian particle"""
    def __init__(self, x, y, material, phase)
    def advect(self, velocity, dt)
    def update_stress(self, strain_rate, dt)

class MarkerSwarm:
    """Collection of markers with batch operations"""
    def advect_all(self, velocity_field, dt)
    def reseed(self, grid)
    def interpolate_to_nodes(self, grid)

class StokesFlow:
    """Main flow solver"""
    def __init__(self, grid, materials, markers)
    def assemble_system(self)
    def solve_picard(self, max_iterations)
    def solve_newton(self, initial_residual)

class Simulation:
    """Orchestrates time stepping"""
    def __init__(self, config)
    def step(self)
    def output(self, iteration)
    
    def run(self, num_iterations):
        for t in range(num_iterations):
            self.step()
            if self.should_output(t):
                self.output(t)
```

### Performance Improvements
1. **Sparse Linear Algebra**: Use `scipy.sparse.linalg` with iterative solvers (GMRes, BiCG)
2. **Numba JIT**: Compile hot loops to machine code
3. **GPU Acceleration**: CuPy for GPU matrix operations
4. **Vectorization**: NumPy broadcasting instead of loops
5. **Multigrid Solvers**: Faster than direct methods for large systems

---

## Part 9: Key Input Parameters (Reference)

```matlab
% Time
Nt = 1600              % number of time steps
dt_out = 20            % output frequency

% Domain
xsize = 170e3          % width (m)
ysize = 60e3           % depth (m)

% Grid spacing (variable resolution)
GRID.dx = [2000, 500, 2000]    % x-spacing (m)
GRID.x = [50e3, 140e3, 170e3]  % x-boundaries for each zone

% Markers
Mquad = 10             % markers per quadrant
Mquad_crit = 5         % minimum for reseeding

% Material
MAT(i).rho0            % reference density
MAT(i).G               % elastic shear modulus
MAT(i).pre_diff/disc   % creep pre-exponentials
MAT(i).Ediff/disc      % activation energies
MAT(i).ndiff/disc      % stress exponents
MAT(i).mu              % friction coefficient
MAT(i).Cmax/min        % cohesion bounds

% Physics
PARAMS.YNElast = 1     % elasticity on/off
PARAMS.YNPlas = 1      % plasticity on/off
PARAMS.Tsolve = 1      % thermal solver on/off

% Solver
PARAMS.Npicard_min = 10        % min Picard iterations
PARAMS.Npicard_max = 100       % max iterations
PARAMS.conv_crit_ResL2 = 1e-9  % convergence tolerance
PARAMS.pitswitch = 0           % switch to Newton iteration

% Stability
PARAMS.fracCFL = 0.5   % CFL fraction for time step
PARAMS.etamax/min      % viscosity bounds
```

---

## Summary: Ready for Speckit Design

Now you have a comprehensive understanding of:

1. **What SiSteR does**: Simulates geodynamic processes with complex rheology on a staggered grid
2. **The physics**: Stokes equations with non-linear, temperature/stress-dependent viscosity
3. **The numerics**: Finite difference discretization, Picard/Newton iteration, sparse matrix solve
4. **The algorithm**: Time-stepping with Lagrangian marker advection and feedback to Eulerian grid
5. **The code structure**: Functional MATLAB with material properties, grid management, and rheology models
6. **OOP opportunities**: Clear separation into Grid, Material, Marker, Rheology, and Solver classes

**Next Steps**: When ready, we can use Speckit to design modular, high-performance Python implementations of these core components.
