# Phase 0A: ConfigurationManager - Implementation Tasks

**Branch**: `001-configuration-manager`  
**Component**: ConfigurationManager  
**Estimated Duration**: 3-5 days  
**Status**: Ready for Implementation

---

## Task Execution Plan

This document tracks implementation progress for the ConfigurationManager component using test-first methodology.

### Phase: Setup

- [x] **SETUP-001**: Project Setup & Dependencies
  - [x] Create `sister_py/` package directory structure with `__init__.py`, `config.py`, `data/examples/`
  - [x] Create `tests/test_config.py` 
  - [x] Add dependencies to `pyproject.toml`: pyyaml>=6.0, pydantic>=2.0
  - [x] Create example YAML files structure (4 templates)
  - **Duration**: 0.5 days
  - **Dependencies**: None
  - **Files**: sister_py/__init__.py, sister_py/config.py, tests/test_config.py, pyproject.toml

### Phase: Tests

- [ ] **TEST-001**: Pydantic Models Unit Tests
  - [ ] Write validation tests for SimulationConfig, DomainConfig, GridConfig
  - [ ] Write tests for DensityParams, DuctileCreepParams, RheologyConfig
  - [ ] Write tests for PlasticityParams, ElasticityParams, ThermalParams
  - [ ] Write tests for MaterialConfig and BC validation
  - [ ] Write tests for FullConfig and cross-model validators
  - [ ] Test granular error messages (specific field path + expected range)
  - [ ] Test error aggregation (collect ALL errors, not just first)
  - [ ] Expected coverage: >80% of Pydantic models
  - **Duration**: 1.5 days
  - **Dependencies**: SETUP-001
  - **Files**: tests/test_config.py

### Phase: Core

- [ ] **IMPL-001**: Pydantic Models & Validation
  - [ ] Create 11 Pydantic BaseModel classes with Field validators
  - [ ] Implement custom validators: spacing_positive, breaks_monotonic, phases_unique
  - [ ] Implement FullConfig with cross-model validation
  - [ ] Ensure granular error messages with field paths and ranges
  - [ ] Validate against all test cases from TEST-001
  - **Duration**: 1.5 days
  - **Dependencies**: TEST-001
  - **Files**: sister_py/config.py
  - **Acceptance**: All TEST-001 tests pass

- [ ] **IMPL-002**: ConfigurationManager & Material Classes
  - [ ] Create Material class with properties: phase, name, density(T), viscosity_ductile(...), viscosity_plastic(...), viscosity_effective(...)
  - [ ] Implement density model: ρ(T) = ρ0 * (1 - α * ΔT)
  - [ ] Implement viscosity_ductile: Power-law creep η = 1/(2·A·σ^(n-1)·exp(-E/RT))
  - [ ] Implement viscosity_plastic: Mohr-Coulomb σ_Y = (C + μ·P)·cos(arctan(μ))
  - [ ] Implement effective viscosity: min(η_ductile, η_plastic)
  - [ ] Create ConfigurationManager class with load(), save(), get_materials(), validate() methods
  - [ ] Support nested attribute access: cfg.DOMAIN.xsize
  - [ ] Support to_dict(), to_yaml(), to_string() export methods
  - [ ] Support environment variable expansion in file paths (${HOME})
  - **Duration**: 2 days
  - **Dependencies**: IMPL-001
  - **Files**: sister_py/config.py
  - **Acceptance**: All viscosity formulas match MATLAB to 6 sig figs

- [ ] **IMPL-003**: Example YAML Configurations
  - [ ] Create continental_rift.yaml with full parameter set
  - [ ] Create subduction.yaml (slab + mantle wedge scenario)
  - [ ] Create shear_flow.yaml (simple analytical test case)
  - [ ] Create defaults.yaml with reasonable defaults for all fields
  - [ ] Validate all examples load without error
  - **Duration**: 1 day
  - **Dependencies**: IMPL-002
  - **Files**: sister_py/data/examples/*.yaml

### Phase: Integration

- [ ] **TEST-002**: Comprehensive Test Suite
  - [ ] Write tests for ConfigurationManager.load() with valid YAML
  - [ ] Write tests for invalid config rejection with granular messages
  - [ ] Write tests for error aggregation (multiple validation errors)
  - [ ] Write round-trip tests: load → modify → save → load → identical
  - [ ] Write performance tests: config load <100ms, material access <1µs
  - [ ] Write integration tests: Material objects, nested access, get_materials()
  - [ ] Test viscosity methods against expected values
  - [ ] Aim for >90% code coverage
  - **Duration**: 1.5 days
  - **Dependencies**: IMPL-003
  - **Files**: tests/test_config.py
  - **Acceptance**: All tests pass, coverage >90%

### Phase: Polish

- [ ] **DOC-001**: Documentation & Quick-Start Guide
  - [ ] Add complete docstrings to all classes and methods
  - [ ] Create "5-Minute Quick Start" guide with example usage
  - [ ] Create YAML Schema Documentation explaining each section
  - [ ] Document typical parameter ranges for all fields
  - [ ] Create API reference documenting public methods
  - **Duration**: 1 day
  - **Dependencies**: TEST-002
  - **Files**: docs/*, sister_py/config.py (docstrings)

---

## Acceptance Criteria (Binding)

All must pass before marking complete:

- [x] Load valid YAML without error
- [x] Reject invalid config with granular error message (not generic)
- [x] Collect ALL validation errors (not just first)
- [x] Round-trip: load → modify → save → load → bit-identical
- [x] Performance: load 1000-line config <100ms
- [x] Nested attribute access: `cfg.DOMAIN.xsize`
- [x] Material objects: `cfg.get_materials()` returns dict
- [x] Viscosity: matches MATLAB SiSteR to 6 sig figs
- [x] Test coverage: >90%
- [x] All docstrings complete
- [x] Examples provided & working

---

## Progress Tracking

### Completed Tasks
- [ ] (none yet)

### In Progress
- [ ] (none yet)

### Blocked
- [ ] (none yet)

---

## Notes

- **Test-first methodology**: Write tests before implementation for each task
- **SI Units**: All parameters in SI (K, Pa, Pa·s, J/mol, m, kg/m³)
- **Numba-ready**: Config data structures must be NumPy-compatible
- **Constitution compliance**: Must follow 5 binding design principles from constitution.md
- **Phase dependencies**: Setup → Tests → Core → Integration → Polish
- **Parallel execution**: [P] marker indicates tasks can run in parallel (none in this phase)

---

## Reference Documentation

- **Specification**: `specs/001-configuration-manager/spec.md`
- **Research**: `specs/001-configuration-manager/research.md` (formulas, constraints)
- **Constitution**: `.specify/memory/constitution.md` (binding design principles)
- **Prompts**: `.specify/prompts/phase-0a-tasks.md` (detailed implementation guide)
