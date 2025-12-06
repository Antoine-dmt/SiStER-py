# Feature Specification: ConfigurationManager

**Feature Branch**: `001-configuration-manager`  
**Created**: 2025-12-06  
**Status**: Ready for Speckit Implementation  
**Input**: YAML-based configuration system for SiSteR-py simulations with full validation and round-trip export

## User Scenarios & Testing

### User Story 1 - Load and Validate Configuration File (Priority: P1)

Users need to load a YAML configuration file and receive immediate feedback if parameters are invalid. This is the critical path for every simulation—without valid config, nothing runs.

**Why this priority**: Without loading configs, the entire system is blocked. This is the foundational capability.

**Independent Test**: Loading a valid YAML file creates a usable ConfigurationManager object without errors.

**Acceptance Scenarios**:

1. **Given** a valid `continental_rift.yaml` file exists, **When** `ConfigurationManager.load("continental_rift.yaml")` is called, **Then** a ConfigurationManager instance is returned with all parameters accessible
2. **Given** an invalid config with `mu=1.5` (out of range), **When** loading is attempted, **Then** a granular error "friction at MATERIALS[1].plasticity.mu = 1.5, expected 0 < μ < 1" is raised
3. **Given** multiple validation errors (negative spacing AND invalid friction), **When** loading is attempted, **Then** ALL errors are collected and reported together (not just first)

---

### User Story 2 - Access Configuration Parameters (Priority: P1)

Users and downstream components (Grid, Material, Solver) need to access config values via intuitive attribute paths without parsing YAML themselves.

**Why this priority**: All other components depend on this interface. Grid needs `cfg.DOMAIN.xsize`, Material needs `cfg.MATERIALS[0].density.rho0`, Solver needs `cfg.SOLVER.conv_tol`.

**Independent Test**: Attribute access returns correct values for all config sections.

**Acceptance Scenarios**:

1. **Given** loaded config, **When** accessing `cfg.DOMAIN.xsize`, **Then** returns 170000.0 (float)
2. **Given** loaded config, **When** accessing `cfg.MATERIALS[0].density.rho0`, **Then** returns 1000.0
3. **Given** loaded config, **When** accessing `cfg.BC['top'].vx`, **Then** returns 1e-10
4. **Given** loaded config, **When** iterating `cfg.MATERIALS`, **Then** returns list of MaterialConfig objects, each with phase, name, density, rheology, etc.

---

### User Story 3 - Create Material Objects (Priority: P1)

Users call `config.get_materials()` to get Material instances that compute viscosity, density, and other rheological properties on demand.

**Why this priority**: Material objects are passed to solver/stress update routines. Without them, no simulation runs.

**Independent Test**: `get_materials()` returns dict of Material objects with working viscosity methods.

**Acceptance Scenarios**:

1. **Given** loaded config, **When** calling `materials = cfg.get_materials()`, **Then** returns dict with int keys (phase IDs) and Material values
2. **Given** materials dict, **When** calling `materials[1].viscosity_ductile(sigma_II=1e7, eps_II=1e-15, T=1200)`, **Then** returns float ≈ 1e20 Pa·s
3. **Given** materials dict, **When** calling `materials[1].density(T=1200)`, **Then** returns ρ using ρ(T) = ρ0·(1 - α·ΔT)

---

### User Story 4 - Export Config for Reproducibility (Priority: P2)

Users save modified configs back to YAML files to document their simulations and enable reproduction.

**Why this priority**: Critical for scientific reproducibility but not blocking simulation execution. Can be implemented after P1 stories.

**Independent Test**: Load YAML → modify → save → reload → configs are bit-identical.

**Acceptance Scenarios**:

1. **Given** loaded config, **When** calling `cfg.to_yaml("output.yaml")`, **Then** file is created with valid YAML syntax
2. **Given** saved YAML file, **When** loading it again with `ConfigurationManager.load("output.yaml")`, **Then** all parameters match original (6 sig figs for floats)
3. **Given** programmatic modification `cfg.SIMULATION.Nt = 100`, **When** calling `cfg.validate()`, **Then** re-validates without error

---

### User Story 5 - Access String Representation (Priority: P3)

Users print or log config values in human-readable format for documentation and debugging.

**Why this priority**: Nice-to-have for documentation. Doesn't block simulations. Implement after P1/P2.

**Acceptance Scenarios**:

1. **Given** loaded config, **When** calling `print(cfg.to_string())`, **Then** returns formatted string showing all sections and values
2. **Given** loaded config, **When** calling `cfg.to_dict()`, **Then** returns nested dict JSON-serializable for APIs/logging

---

### Edge Cases

- What happens when environment variable `${HOME}` is referenced but HOME is not set? → Raise clear error
- What happens when included YAML file (`!include`) references non-existent file? → Raise clear error
- What happens when grid zone boundaries overlap? → Raise clear error
- What happens when user modifies config after loading and before calling validate()? → Should re-validate successfully or fail with clear message

## Requirements

### Functional Requirements

- **FR-001**: ConfigurationManager MUST load YAML files using pyyaml and parse multi-line strings, nested dicts, lists
- **FR-002**: ConfigurationManager MUST validate all parameters against Pydantic v2 BaseModel schema (SIMULATION, DOMAIN, GRID, MATERIALS, BC, PHYSICS, SOLVER)
- **FR-003**: ConfigurationManager MUST collect ALL validation errors before raising (not just first error)
- **FR-004**: Validation error messages MUST be granular and show path and value: "friction at MATERIALS[1].plasticity.mu = 1.5, expected 0 < μ < 1"
- **FR-005**: ConfigurationManager MUST provide nested attribute access: `cfg.DOMAIN.xsize`, `cfg.MATERIALS[0].density.rho0`
- **FR-006**: ConfigurationManager MUST support environment variable substitution: `${HOME}/data/` expands to actual path
- **FR-007**: ConfigurationManager MUST support YAML includes: `!include "base_config.yaml"` loads other files
- **FR-008**: ConfigurationManager MUST preserve YAML comments for round-trip export
- **FR-009**: ConfigurationManager MUST provide `get_materials()` method returning dict[int, Material] with viscosity methods
- **FR-010**: ConfigurationManager MUST provide `to_yaml()`, `to_dict()`, `to_string()`, `validate()` export methods
- **FR-011**: Material objects MUST have methods: `viscosity_ductile()`, `viscosity_plastic()`, `viscosity_effective()`, `density(T)`
- **FR-012**: Viscosity computations MUST use power-law rheology: ε̇ = A·σⁿ·exp(-E/RT)
- **FR-013**: All parameters MUST be in SI units (Pa, Pa·s, K, J/mol, m, kg/m³)
- **FR-014**: Temperature MUST be Kelvin, never Celsius
- **FR-015**: Friction coefficient MUST be constrained 0 < μ < 1
- **FR-016**: Grid spacing MUST be positive and monotonic increasing at zone boundaries
- **FR-017**: Phase IDs MUST be unique among materials

### Key Entities

- **ConfigurationManager**: Loads, validates, provides access to all config parameters
  - Methods: `load(filepath)`, `get_materials()`, `to_yaml()`, `to_dict()`, `to_string()`, `validate()`
  - Attributes: `SIMULATION`, `DOMAIN`, `GRID`, `MATERIALS`, `BC`, `PHYSICS`, `SOLVER` (all nested)

- **Material**: Wraps material properties and computes rheology on demand
  - Methods: `viscosity_ductile()`, `viscosity_plastic()`, `viscosity_effective()`, `density(T)`
  - Attributes: phase, name, density (rho0, alpha), rheology, plasticity, elasticity, thermal

- **Pydantic BaseModels**: SimulationConfig, DomainConfig, GridConfig, MaterialConfig, BCConfig, PhysicsConfig, SolverConfig
  - Each has validators for range checks, monotonicity, uniqueness

## Success Criteria

### Measurable Outcomes

- **SC-001**: Config load performance: 1000-line YAML file loads in < 100 ms
- **SC-002**: Validation error granularity: all error messages show parameter path, value, and expected range
- **SC-003**: Round-trip fidelity: load YAML → modify → save → reload → bit-identical to original (6 sig figs for floats)
- **SC-004**: Comment preservation: YAML comments preserved after round-trip export
- **SC-005**: Material property accuracy: viscosity_ductile() matches MATLAB SiSteR to 6 significant figures for identical inputs
- **SC-006**: Attribute access works: nested access like `cfg.MATERIALS[0].density.rho0` returns correct value
- **SC-007**: Material object creation: `get_materials()` returns dict with all 2 phases and methods callable
- **SC-008**: Test coverage: > 90% code coverage for config.py
- **SC-009**: Validation completeness: 100% of validation errors collected before raising (no early exit)
- **SC-010**: User accessibility: users can copy example YAML, modify 3-4 parameters, and run without code changes

### Example Validation Scenario

**Given** a config file with three validation errors:
- `mu=1.5` (friction out of range)
- `x_spacing[1]=-500` (negative grid spacing)
- `Nt=-100` (negative time steps)

**When** loading is attempted, **Then** error should list ALL THREE:
```
ValidationError: 3 errors found:
  1. friction at MATERIALS[0].plasticity.mu = 1.5, expected 0 < μ < 1
  2. Grid spacing must be positive, got -500 at GRID.x_spacing[1]
  3. Nt must be positive, got -100 at SIMULATION.Nt
```

Not just the first error.
