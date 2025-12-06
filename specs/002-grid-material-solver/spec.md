# Phase 1: Grid, Material & Solver Modules

**Workflow**: speckit.specify  
**Component**: Phase 1 (Grid, Material, Solver)  
**Status**: READY FOR SPECKIT  
**Date**: December 6, 2025  

---

## Overview

Phase 1 extends Phase 0A (ConfigurationManager) with three core computational modules:
- **Grid Module**: Generate fully-staggered grids with variable spacing
- **Material Module**: Interpolate material properties to grid nodes
- **Solver Module**: Assemble Stokes system with boundary conditions

These three modules form the computational kernel for geodynamic simulations.

---

## User Stories

### P1: Grid Generation for Geodynamic Simulations

**As a** geodynamic modeler  
**I want** to generate a fully-staggered grid from configuration  
**So that** I can solve Stokes equations with minimal errors  

**Acceptance Criteria**:
- Grid generates from cfg.GRID configuration
- Supports variable spacing per zone (cfg.GRID.x_breaks, cfg.GRID.y_breaks)
- Returns both normal nodes (x_n, y_n) and staggered nodes (x_s, y_s)
- Grid spacing validated against dx/dy constraints
- Performance: Generate 500×500 grid in <1 second

### P1: Material Properties on Grid Nodes

**As a** physics simulator  
**I want** material properties interpolated to all grid nodes  
**So that** I can use them in Stokes solver  

**Acceptance Criteria**:
- Load Material objects from ConfigurationManager
- Interpolate to normal nodes (velocity, pressure)
- Interpolate to staggered nodes (shear stress)
- Support all property types: viscosity, density, elasticity
- Performance: Interpolate 500×500 grid in <100ms

### P1: Stokes System Assembly

**As a** numerical analyst  
**I want** assembled Stokes matrix with boundary conditions  
**So that** I can pass to linear solver  

**Acceptance Criteria**:
- Assemble Stokes operator (−∇²u, ∇P terms)
- Apply velocity boundary conditions
- Apply pressure boundary conditions
- Support traction (stress) boundary conditions
- Performance: Assemble 500×500 system in <500ms

---

## Functional Requirements

### Grid Module (FR 1-6)

**FR-1**: Generate coordinate arrays from zone configuration
- Input: cfg.GRID with x_breaks, y_breaks, x_spacing, y_spacing
- Output: Coordinate arrays for normal and staggered nodes
- Constraint: Zone boundaries must be strictly increasing

**FR-2**: Support fully-staggered grid discretization
- Normal nodes: (xn[i], yn[j])
- Staggered nodes X: (xs[i], yn[j])
- Staggered nodes Y: (xn[i], ys[j])
- Compliance: Duretz et al. (2013) standard

**FR-3**: Validate grid spacing constraints
- x_spacing > 0, y_spacing > 0
- All cells have positive width and height
- Max aspect ratio configurable (default 1.5)
- Boundary cells follow zone discretization

**FR-4**: Generate visualization-ready output
- Return grid as dict with keys: x_n, y_n, x_s, y_s
- Include node counts and cell dimensions
- Metadata: total cells, total nodes, max/min cell size

**FR-5**: Support grid refinement patterns
- Uniform spacing (single value)
- Linear refinement (coarse → fine → coarse)
- Exponential refinement (for boundary layers)
- Custom breakpoint sequences

**FR-6**: Interface with ConfigurationManager
- Accept cfg object from Phase 0A
- Extract domain and grid parameters
- Validate grid dimensions against domain size
- Return Grid object with coordinate access

### Material Module (FR 7-12)

**FR-7**: Load material properties at grid nodes
- Input: Material objects from cfg.get_materials()
- Input: Grid object from Grid module
- Output: Material properties on all nodes

**FR-8**: Interpolate to normal nodes
- Properties: viscosity, density, elasticity
- Method: Marker-in-cell or arithmetic averaging
- Preserve phase boundaries where possible
- Handle material discontinuities smoothly

**FR-9**: Interpolate to staggered nodes
- Shear stress nodes (x_s, yn) and (xn, y_s)
- Average across cell faces
- Maintain rheology consistency
- Support anisotropic properties

**FR-10**: Evaluate material-dependent properties
- Density: ρ(T, P) with depth correction
- Viscosity: η(σ, ε, T) from power-law creep
- Elastic moduli: K, G from elasticity parameters
- Plasticity: cohesion, friction from depth

**FR-11**: Create MaterialGrid object
- dict-like access: matgrid['viscosity'][i,j]
- Property arrays: (n_nodes_x, n_nodes_y) or (n_cells_x, n_cells_y)
- Metadata: min/max values, phase distribution
- Export to NumPy for downstream operations

**FR-12**: Interface with Grid and ConfigurationManager
- Accept Grid object from Grid module
- Accept cfg object from Phase 0A
- Return MaterialGrid dict ready for Solver

### Solver Module (FR 13-18)

**FR-13**: Assemble Stokes operator matrix
- Build Laplacian term for velocity (−∇²u)
- Build divergence term (∇·u) for continuity
- Build gradient term (∇P) for pressure coupling
- Sparse matrix format (scipy.sparse)

**FR-14**: Apply velocity boundary conditions
- Dirichlet BC: u = prescribed velocity
- Essential BC: Set rows in velocity system
- Zero velocity: u = 0 (no-slip)
- Analytical BC: u(x,y) from formula

**FR-15**: Apply pressure boundary conditions
- Neumann BC: ∂u/∂n = prescribed stress
- Natural BC: Integrated into variational form
- Free surface: Dynamic pressure boundary
- Hydrostatic reference pressure

**FR-16**: Apply traction boundary conditions
- Shear stress: τ_xz, τ_yz prescribed
- Normal stress: σ_xx, σ_yy prescribed
- Mixed BC: u on some edges, τ on others
- Stress-velocity conversion at boundaries

**FR-17**: Create Stokes system object
- Matrix A: (n_dof, n_dof) sparse matrix
- RHS vector b: (n_dof,) for forcing
- DOF mapping: velocity and pressure indices
- Solver metadata: condition number, matrix properties

**FR-18**: Interface with Grid and MaterialGrid
- Accept Grid from Grid module
- Accept MaterialGrid from Material module
- Accept cfg.SOLVER, cfg.BC, cfg.PHYSICS configs
- Return SolverSystem ready for linear algebra

---

## Success Criteria

### SC-1: Grid Module Correctness
- Generated grid matches analytical solution for uniform spacing
- Zone boundaries align with cfg.GRID.x_breaks/y_breaks
- Staggered node positions correct per Duretz et al. (2013)
- All coordinate arrays have correct shape and monotonicity

### SC-2: Grid Module Performance
- Generate 500×500 uniform grid in <500ms
- Generate 500×500 refined grid in <1s
- Memory usage <100MB for 500×500 grid
- No redundant array allocations

### SC-3: Material Interpolation Accuracy
- Interpolated viscosity matches analytical profile (exponential test)
- Phase boundaries resolved to within 1 cell width
- Material properties conserve mass (density integral)
- Temperature field interpolates smoothly

### SC-4: Material Module Performance
- Interpolate properties for 500×500 grid in <100ms
- Load material from YAML and interpolate in <200ms total
- Memory efficient: store only used properties
- Vectorized NumPy operations (no Python loops)

### SC-5: Stokes Assembly Correctness
- Assembled matrix satisfies variational principle
- Test case: Analytical Poiseuille flow yields exact solution
- Boundary conditions applied consistently
- Divergence constraint preserved (∇·u = 0)

### SC-6: Stokes Assembly Performance
- Assemble 500×500 system in <500ms
- System solve time (with direct solver) <1s
- Memory: sparse matrix uses <500MB for 500×500
- Condition number <1e6 for typical rheology

### SC-7: Integration with Phase 0A
- cfg = ConfigurationManager.load('config.yaml')
- grid = Grid.generate(cfg)
- matgrid = MaterialGrid.load(cfg, grid)
- solver = SolverSystem.assemble(cfg, grid, matgrid)
- All operations chain without data conversion

### SC-8: Code Quality
- >90% test coverage for all three modules
- Comprehensive docstrings (100%)
- Type annotations throughout
- 0 Pylint warnings (with reasonable config)

### SC-9: Documentation
- API reference for all public classes/methods
- Quickstart examples for each module
- Theory section explaining discretization
- Troubleshooting guide for common issues

### SC-10: Examples
- continental_rift_phase1.yaml (complete workflow)
- subduction_phase1.yaml (complex domain)
- analytical_test.yaml (validation)
- All examples run end-to-end

---

## Technical Constraints

### Architecture
- **Three separate modules** in sister_py/: grid.py, material.py, solver.py
- **Dependency chain**: Grid → Material → Solver
- **Interface**: Each module accepts cfg + upstream outputs
- **Output format**: NumPy arrays or dicts for efficiency

### Physics
- **Grid**: Fully-staggered (Duretz et al. 2013 standard)
- **Material**: Power-law creep with temperature/pressure effects
- **Solver**: Incompressible Stokes equations (∇·u = 0)
- **Units**: SI throughout (m, s, Pa, K)

### Performance
- **Grid generation**: <1 second for 500×500 grid
- **Material interpolation**: <100ms for full interpolation
- **Stokes assembly**: <500ms for 500×500 system
- **Total Phase 1 initialization**: <2 seconds

### Compatibility
- **Python**: 3.10+ (same as Phase 0A)
- **Dependencies**: NumPy, SciPy, Pydantic (existing)
- **Numba**: Optional for future @njit optimization
- **Data format**: NumPy arrays for downstream processing

---

## Implementation Phases

### Phase 1A: Grid Module (3-4 days)
1. Core: Coordinate generation with zone support
2. Validation: Spacing constraints, monotonicity
3. Testing: 40+ test cases, unit + integration
4. Documentation: API reference + quickstart
5. Examples: 2 grid examples (uniform + refined)

### Phase 1B: Material Module (3-4 days)
1. Core: Property interpolation to nodes
2. Validation: Accuracy against analytical profiles
3. Testing: 35+ test cases, interpolation accuracy
4. Documentation: API reference + physics explanation
5. Examples: 2 material examples (single + multi-phase)

### Phase 1C: Solver Module (4-5 days)
1. Core: Stokes matrix assembly
2. Validation: Poiseuille flow analytical test
3. Testing: 45+ test cases, BC application
4. Documentation: API reference + discretization theory
5. Examples: 3 solver examples (simple + complex)

**Total Phase 1**: 10-13 days (120+ test cases)

---

## Dependencies & Handoff

### Requires from Phase 0A
- ✅ ConfigurationManager.load(filepath)
- ✅ cfg.DOMAIN.xsize, cfg.DOMAIN.ysize
- ✅ cfg.GRID.x_spacing, cfg.GRID.x_breaks, etc.
- ✅ cfg.MATERIALS, cfg.SOLVER, cfg.PHYSICS, cfg.BC
- ✅ Material objects with property methods

### Provides to Phase 2
- Grid coordinates and node mappings
- Material properties on nodes (interpolated)
- Assembled Stokes system (A, b)
- Output interface for next module

### Testing Strategy
- **Unit tests**: Each function in isolation
- **Integration tests**: Full workflows (cfg → grid → material → solver)
- **Regression tests**: Analytical solution validation
- **Performance tests**: Timing and memory profiling

---

## Acceptance & Sign-Off

### Pre-Development
- [ ] Specification reviewed and approved
- [ ] Architecture diagram validated
- [ ] Test strategy confirmed
- [ ] Performance targets agreed

### Post-Implementation
- [ ] All 120+ tests passing
- [ ] >90% code coverage
- [ ] Documentation complete
- [ ] Examples working end-to-end
- [ ] Performance targets met
- [ ] Code review passed
- [ ] Ready for Phase 2

---

## References

### Fully-Staggered Grid
Duretz, C., et al. (2013). "Discretization errors and free surface stability in the finite difference and marker-in-cell method." *Journal of Computational Physics*, 249, 127-140.

### Stokes Equations
Gerya, T. (2010). "Introduction to Numerical Geodynamic Modeling." *Cambridge University Press*.

### Rheology Integration
Hirth, G., & Kohlstedt, D. (2003). "Rheology of the upper mantle and the mantle wedge: A view from the experimentalists." *Geophysical Monograph Series*.

---

**Specification Status**: READY FOR DEVELOPMENT  
**Component**: Phase 1 (Grid, Material, Solver)  
**Scope**: 3 modules, 18 FRs, 10 SCs, 120+ tests  
**Effort**: 10-13 days estimated
