# Phase 1 Planning Document

**Component**: Grid, Material & Solver Modules  
**Workflow**: speckit.plan  
**Status**: PLANNING  
**Date**: December 6, 2025  

---

## Project Context

**SiSteR-py** is a Python port of the MATLAB geodynamic simulator SiSteR.

### Architecture Overview

```
Phase 0A: ConfigurationManager ✅ COMPLETE
    ↓ (cfg object)
Phase 1A: Grid Module (NEW)
    ↓ (grid coordinates)
Phase 1B: Material Module (NEW)
    ↓ (properties on nodes)
Phase 1C: Solver Module (NEW)
    ↓ (assembled system)
Phase 2: Time-stepping & Output (FUTURE)
```

### Core Physics

- **Stokes equations**: Incompressible viscous flow
- **Power-law creep**: Temperature and stress-dependent viscosity
- **Mohr-Coulomb plasticity**: Yield criterion and strain weakening
- **Thermal effects**: Density and viscosity depend on temperature

---

## Technical Stack

### Required Technologies
- **Python 3.10+**: Language requirement
- **NumPy**: Array operations, linear algebra basics
- **SciPy**: Sparse matrices, linear solvers
- **Pydantic**: Configuration validation (from Phase 0A)

### Development Tools
- **pytest**: Unit and integration testing
- **pytest-cov**: Code coverage measurement
- **black**: Code formatting
- **ruff**: Linting and static analysis
- **mypy**: Type checking

### Optional (Future)
- **Numba**: @njit compilation for 50x speedup
- **sparse**: Alternative sparse matrix format
- **h5py**: HDF5 output (Phase 2+)

---

## File Structure

### Phase 1A: Grid Module

```
sister_py/
├── grid.py                    [300-400 lines]
│   ├── Grid (class)
│   ├── GridConfig support
│   └── Coordinate generation
└── data/
    └── examples/
        ├── grid_uniform.yaml
        └── grid_refined.yaml

tests/
└── test_grid.py              [400-500 lines]
    ├── TestUniformGrid
    ├── TestRefinedGrid
    ├── TestGridValidation
    └── TestGridPerformance

docs/
└── GRID_GUIDE.md             [300-400 lines]
    ├── API reference
    ├── Coordinate system
    ├── Staggered grid explanation
    └── Examples
```

### Phase 1B: Material Module

```
sister_py/
├── material_grid.py           [300-400 lines]
│   ├── MaterialGrid (class)
│   ├── Interpolation logic
│   └── Property evaluation
└── data/
    └── examples/
        ├── material_simple.yaml
        └── material_complex.yaml

tests/
└── test_material.py           [400-500 lines]
    ├── TestMaterialInterp
    ├── TestPhaseDistribution
    ├── TestPropertyAccuracy
    └── TestMaterialPerformance

docs/
└── MATERIAL_GUIDE.md          [300-400 lines]
    ├── Interpolation methods
    ├── Property calculations
    ├── Phase handling
    └── Examples
```

### Phase 1C: Solver Module

```
sister_py/
├── solver.py                  [400-500 lines]
│   ├── SolverSystem (class)
│   ├── Stokes assembly
│   ├── BC application
│   └── Matrix construction
└── data/
    └── examples/
        ├── solver_simple.yaml
        ├── solver_complex.yaml
        └── solver_validation.yaml

tests/
└── test_solver.py             [500-600 lines]
    ├── TestStokesBuild
    ├── TestBCApplication
    ├── TestAnalyticalValidation
    └── TestSolverPerformance

docs/
└── SOLVER_GUIDE.md            [400-500 lines]
    ├── Stokes discretization
    ├── Boundary condition types
    ├── System assembly process
    └── Examples
```

### Total New Code
- **sister_py/*.py**: ~1,000-1,300 lines (3 modules)
- **tests/test_*.py**: ~1,300-1,600 lines (120+ tests)
- **docs/**: ~1,000-1,200 lines (3 guides)
- **examples/**: ~500-700 lines (8-9 YAML files)
- **Total**: ~3,800-4,400 lines of code + tests + docs

---

## Development Plan

### Phase 1A: Grid Module (3-4 days)

**Day 1: Core Implementation**
- Implement Grid class with configuration loading
- Coordinate generation for uniform spacing
- Zone-based breakpoint handling
- Basic validation (spacing > 0, bounds checking)

**Day 2: Advanced Features**
- Staggered node positioning (xs, ys arrays)
- Coordinate array validation (monotonicity)
- Variable spacing per zone support
- Visualization metadata generation

**Day 3: Testing & Documentation**
- 40+ unit and integration tests
- >90% code coverage
- GRID_GUIDE.md documentation
- 2 example configurations

**Day 4 (Optional): Polish**
- Performance optimization
- Error message improvement
- Code review feedback integration

**Key Outputs**:
- ✅ `sister_py/grid.py` (300-400 lines)
- ✅ `tests/test_grid.py` (400-500 lines)
- ✅ `docs/GRID_GUIDE.md` (300-400 lines)
- ✅ 2 example YAML files

### Phase 1B: Material Module (3-4 days)

**Day 1: Interpolation Core**
- MaterialGrid class structure
- Integration with Grid and ConfigurationManager
- Basic interpolation to normal nodes (arithmetic mean)
- Property array construction

**Day 2: Advanced Interpolation**
- Staggered node interpolation
- Phase boundary handling
- Material discontinuity smoothing
- Metadata and diagnostics

**Day 3: Testing & Documentation**
- 35+ unit and integration tests
- Accuracy validation against analytical profiles
- MATERIAL_GUIDE.md documentation
- 2 example configurations

**Day 4 (Optional): Polish**
- Interpolation method alternatives (marker-in-cell)
- Performance optimization
- Edge case handling

**Key Outputs**:
- ✅ `sister_py/material_grid.py` (300-400 lines)
- ✅ `tests/test_material.py` (400-500 lines)
- ✅ `docs/MATERIAL_GUIDE.md` (300-400 lines)
- ✅ 2 example YAML files

### Phase 1C: Solver Module (4-5 days)

**Day 1: Matrix Assembly**
- SolverSystem class structure
- Laplacian operator assembly
- Continuity equation integration
- Sparse matrix construction (scipy.sparse)

**Day 2: Boundary Conditions**
- Velocity BC (Dirichlet) implementation
- Pressure BC (Neumann) implementation
- Traction BC handling
- BC node mapping and row modification

**Day 3: System Construction**
- RHS vector generation
- System matrix finalization
- Metadata and diagnostics
- Condition number calculation

**Day 4: Testing & Documentation**
- 45+ unit and integration tests
- Analytical validation (Poiseuille flow)
- SOLVER_GUIDE.md documentation
- 3 example configurations

**Day 5 (Optional): Polish**
- Sparse matrix optimization
- Error handling robustness
- Performance tuning

**Key Outputs**:
- ✅ `sister_py/solver.py` (400-500 lines)
- ✅ `tests/test_solver.py` (500-600 lines)
- ✅ `docs/SOLVER_GUIDE.md` (400-500 lines)
- ✅ 3 example YAML files

---

## Testing Strategy

### Unit Tests (60 tests across 3 modules)

**Grid Module (20 tests)**:
- `test_uniform_grid_generation` - Coordinates match analytical
- `test_zone_breakpoints` - Boundaries align with config
- `test_staggered_nodes` - Positioning per Duretz et al.
- `test_spacing_validation` - Positive spacing enforced
- `test_boundary_cells` - Edge cells correct dimensions
- (15 more edge cases and error conditions)

**Material Module (20 tests)**:
- `test_interpolation_uniform` - Uniform material interpolates correctly
- `test_interpolation_layered` - Layered structure preserved
- `test_phase_boundaries` - Sharp phase transitions handled
- `test_property_ranges` - Min/max values reasonable
- `test_staggered_interpolation` - Staggered nodes correct
- (15 more edge cases and accuracy tests)

**Solver Module (20 tests)**:
- `test_laplacian_assembly` - Operator structure correct
- `test_continuity_equation` - Divergence handled correctly
- `test_dirichlet_bc` - Velocity BC applied correctly
- `test_neumann_bc` - Traction BC integrated
- `test_system_shape` - Matrix dimensions correct
- (15 more tests for BC types and edge cases)

### Integration Tests (30 tests)

- Full workflow: cfg → grid → material → solver
- Round-trip: load config, run all steps, verify output
- Analytical validation: Poiseuille flow exact solution
- Performance: Timing constraints verified
- Memory profiling: No memory leaks

### Regression Tests (30 tests)

- Examples: All 8-9 example configurations
- Cross-module: Grid used by Material, Material by Solver
- Reproducibility: Same config yields same results
- Version compatibility: Works with Phase 0A outputs

**Total Tests**: 120+ across all three modules

---

## Validation Strategy

### Analytical Tests

**Poiseuille Flow** (Solver validation):
- Analytical solution: u(y) = (−∂P/∂x) · y(H−y) / (2η)
- Domain: 100×100 km, uniform viscosity
- BC: Pressure gradient driving flow
- Validation: Computed velocity matches analytical to <1% error

**Exponential Density Profile** (Material validation):
- Analytical: ρ(z) = ρ₀ · exp(−z/H)
- Temperature-driven density variation
- Validation: Interpolated profile matches analytical

**Uniform Grid** (Grid validation):
- Analytical: Uniform spacing Δx = L/nx
- All zones same size, monotonic coordinates
- Validation: Generated coordinates match analytical

### Numerical Tests

- Grid convergence study (error vs grid size)
- Material property conservation (mass integral)
- Stokes solver residual < 1e-10
- Condition number reasonable for typical rheology

### Performance Tests

- Grid generation time < 1 second
- Material interpolation < 100 ms
- Stokes assembly < 500 ms
- Memory usage < 500 MB for 500×500 grid

---

## Code Quality Standards

### Coverage
- **Target**: >90% coverage (all 3 modules)
- **Method**: pytest with pytest-cov
- **Exclusions**: Debug code, platform-specific paths

### Formatting
- **Tool**: black (100 char line length)
- **Enforcement**: Pre-commit hook
- **Exceptions**: None (strict adherence)

### Linting
- **Tool**: ruff (strict mode)
- **Rules**: Follow PEP 8, PEP 257
- **Exclusions**: Long math equations in docstrings

### Type Checking
- **Tool**: mypy (strict mode)
- **Target**: 100% type annotations
- **Allowances**: Generic types where necessary

### Documentation
- **Docstrings**: 100% (all functions, classes, methods)
- **Format**: NumPy-style docstrings
- **Content**: Description, parameters, returns, examples

---

## Dependencies

### Required (Already in Phase 0A)
```toml
pydantic = ">=2.0"
pyyaml = ">=6.0"
numpy = ">=1.21"
scipy = ">=1.7"
```

### Development
```toml
pytest = "^7.0"
pytest-cov = "^4.0"
black = "^23.0"
ruff = "^0.1"
mypy = "^1.0"
```

### Optional (Future)
```toml
numba = ">=0.57"  # For @njit optimization
h5py = ">=3.0"    # For HDF5 output
```

---

## Risk & Mitigation

### Risk 1: Sparse Matrix Performance
**Risk**: Large sparse matrices slow to assemble or solve  
**Mitigation**: Use scipy.sparse, profile before optimization  
**Backup**: Implement CSR format conversion option

### Risk 2: Interpolation Accuracy
**Risk**: Material interpolation produces unphysical properties  
**Mitigation**: Analytical validation tests, bounds checking  
**Backup**: Alternative interpolation methods available

### Risk 3: BC Implementation Complexity
**Risk**: Boundary condition logic hard to debug  
**Mitigation**: Extensive unit tests per BC type  
**Backup**: Simplified BC handler for common cases

### Risk 4: Performance Targets
**Risk**: Assembly/interpolation too slow for large grids  
**Mitigation**: NumPy vectorization, avoid Python loops  
**Backup**: Numba @njit in Phase 1.5

---

## Success Criteria Checklist

### Implementation Complete
- [ ] 3 modules implemented (grid.py, material_grid.py, solver.py)
- [ ] 120+ tests passing
- [ ] >90% code coverage
- [ ] Performance targets met (<1s, <100ms, <500ms)

### Documentation Complete
- [ ] API reference for all classes/methods
- [ ] Theory sections (discretization, BC application)
- [ ] Quickstart examples (3 guides × 2-3 examples each)
- [ ] Troubleshooting guides

### Integration Complete
- [ ] Grid module works with Phase 0A ConfigurationManager
- [ ] Material module integrates with Grid module
- [ ] Solver module integrates with Material module
- [ ] End-to-end workflow (cfg → grid → material → solver)

### Quality Assurance
- [ ] Code review passed
- [ ] All tests passing (100%)
- [ ] No Pylint warnings (reasonable config)
- [ ] mypy strict mode: 0 errors

### Ready for Phase 2
- [ ] Branch 002-grid-material-solver committed
- [ ] Git history clean and logical
- [ ] Documentation complete
- [ ] Examples working end-to-end

---

## Timeline

**Phase 1A (Grid)**: Dec 7-10, 2025 (3-4 days)  
**Phase 1B (Material)**: Dec 11-14, 2025 (3-4 days)  
**Phase 1C (Solver)**: Dec 15-19, 2025 (4-5 days)  
**Total Phase 1**: Dec 7-19, 2025 (~12 days)

**Next**: Phase 2 (Output, TimeStepper) - Jan 2026

---

**Planning Status**: READY FOR IMPLEMENTATION  
**Estimated Effort**: 10-13 days  
**Estimated Lines of Code**: 3,800-4,400  
**Estimated Tests**: 120+  
**Target Completion**: December 19, 2025
