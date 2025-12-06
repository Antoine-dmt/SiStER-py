---
agent: speckit.plan
---

# Phase 0A: ConfigurationManager - Planning Document

**Feature**: ConfigurationManager  
**Branch**: `001-configuration-manager`  
**Specification**: `specs/001-configuration-manager/spec.md`  
**Constitution**: `.specify/memory/constitution.md`  
**Status**: Ready for Planning  

---

## Overview

**What we're building**: A production-grade YAML configuration loader for SiSteR-py geodynamic simulations.

**Why it matters**: ConfigurationManager is the **foundational component**—all other modules (Grid, Material, Solver, TimeStepper) depend on it. One YAML file drives the entire simulation.

**Key constraints**:
- Pydantic v2 for validation (granular error messages)
- SI units throughout (K, Pa, Pa·s, J/mol, m, kg/m³)
- Round-trip fidelity (load → modify → save → load → identical)
- Performance < 100 ms config load
- Test coverage > 90%
- Constitution compliance (5 binding design principles)

---

## 1. Delivery Scope

### 1.1 What's Included (In-Scope)

✅ **ConfigurationManager class**
- Load YAML files with environment variable substitution
- Full validation using Pydantic v2
- Nested attribute access (`cfg.DOMAIN.xsize`)
- Nested + indexing access (`cfg.MATERIALS[0].density.rho0`)
- Export to YAML (with comments, 6 sig figs)
- Export to dict (JSON-serializable)
- Export to string (human-readable)
- Re-validation after programmatic changes

✅ **Material class**
- Wrapper for material properties (validated at config load)
- Viscosity methods:
  - `viscosity_ductile(sigma_II, eps_II, T)` — Power-law creep
  - `viscosity_plastic(sigma_II, P)` — Mohr-Coulomb yield
  - `viscosity_effective(sigma_II, eps_II, T, P)` — Combined
- Density method: `density(T)` — Linear thermal expansion
- All methods SI units, Numba-compatible (no Python objects)

✅ **Pydantic validation models** (11 classes)
- `SimulationConfig`, `DomainConfig`, `GridConfig`
- `DensityParams`, `DuctileCreepParams`
- `RheologyConfig`, `PlasticityParams`, `ElasticityParams`, `ThermalParams`
- `MaterialConfig`, `BCConfig`, `PhysicsConfig`, `SolverConfig`
- `FullConfig` (root)

✅ **Granular error messages**
- Parameter path + value + expected range
- All errors collected (not just first)
- Example: `"friction at MATERIALS[1].plasticity.mu = 1.5, expected 0 < μ < 1"`

✅ **Example YAML files** (4 total)
- `continental_rift.yaml` — Continental rifting benchmark
- `subduction.yaml` — Subduction zone with slab + wedge
- `shear_flow.yaml` — Analytical validation test
- `defaults.yaml` — Default parameter values

✅ **Comprehensive test suite**
- Unit tests (Pydantic validation)
- Round-trip tests (load → modify → save → load)
- Performance tests (< 100 ms, < 1 μs per call)
- Integration tests (Material objects, nested access)
- Edge case tests (invalid paths, missing env vars, overlapping zones)
- **Target coverage: > 90%**

✅ **Documentation**
- Full API docstrings (ConfigurationManager, Material, all Pydantic models)
- "5-Minute Quick Start" guide
- YAML schema documentation (each section explained)
- Example usage patterns
- Troubleshooting guide

### 1.2 What's Excluded (Out-of-Scope)

❌ **Grid implementation** (Phase 1A)  
❌ **Solver** (Phase 3A)  
❌ **Time stepper** (Phase 4A)  
❌ **Marker swarm** (Phase 1C)  
❌ **Thermal solver** (Phase 5A)  
❌ **GPU acceleration** (Phase 6A)  

---

## 2. Success Criteria (Binding)

### 2.1 Functional Requirements

All must pass:

| ID | Requirement | Measurable Outcome |
|----|-------------|-------------------|
| FR-001 | Load valid YAML | No exceptions, ConfigurationManager returned |
| FR-002 | Validate all parameters | Pydantic BaseModel enforces constraints |
| FR-003 | Granular errors | Error message includes parameter path + value + range |
| FR-004 | Collect all errors | Multiple validation errors reported together |
| FR-005 | Nested attribute access | `cfg.DOMAIN.xsize` returns float without error |
| FR-006 | Nested + indexing | `cfg.MATERIALS[0].density.rho0` returns float |
| FR-007 | Create Material objects | `cfg.get_materials()` returns dict[int, Material] |
| FR-008 | Material.density(T) | Returns ρ = ρ₀(1 - α·ΔT) in kg/m³ |
| FR-009 | Material.viscosity_ductile() | Returns η from power-law creep in Pa·s |
| FR-010 | Material.viscosity_plastic() | Returns η from Mohr-Coulomb in Pa·s |
| FR-011 | Material.viscosity_effective() | Returns min(η_ductile, η_plastic) in Pa·s |
| FR-012 | Export to YAML | File saved, preserves comments, 6 sig figs |
| FR-013 | Export to dict | JSON-serializable, all types correct |
| FR-014 | Export to string | Human-readable format for stdout |
| FR-015 | Round-trip fidelity | Load → modify → save → load → identical |
| FR-016 | Re-validate | `cfg.validate()` catches errors in modified config |
| FR-017 | Environment vars | `${HOME}/configs/rift.yaml` expands correctly |

### 2.2 Success Criteria (Measurable)

| ID | Criterion | Target | Verification |
|----|-----------|--------|--------------|
| SC-001 | Config load time | < 100 ms | Benchmark 1000-line YAML |
| SC-002 | Material access | < 1 μs per call | Benchmark 10,000 calls |
| SC-003 | Test coverage | > 90% | pytest --cov=sister_py/config.py |
| SC-004 | All tests pass | 100% | pytest exit code 0 |
| SC-005 | Viscosity accuracy | 6 sig figs vs MATLAB | Compare 10 test cases |
| SC-006 | Error message clarity | Users understand error | Manual review of 5+ error messages |
| SC-007 | Documentation completeness | API reference complete | All public methods have docstrings |
| SC-008 | Example coverage | All features demonstrated | continental_rift, subduction, shear_flow work |
| SC-009 | Edge case handling | No exceptions | Test invalid path, missing env var, overlapping zones |
| SC-010 | Constitution compliance | All 5 principles satisfied | Design review checklist |

---

## 3. Work Breakdown Structure (WBS)

### Phase 0A has 6 major work packages:

```
Phase 0A: ConfigurationManager
├── WP-01: Project Setup
├── WP-02: Data Validation Layer (Pydantic)
├── WP-03: ConfigurationManager & Material Classes
├── WP-04: Example Configurations
├── WP-05: Test Suite
└── WP-06: Documentation & Examples
```

---

## 4. Detailed Work Packages

### WP-01: Project Setup & Dependencies

**Objective**: Establish project structure and dependencies

**Deliverables**:
1. Python package structure: `sister_py/` directory
2. `pyproject.toml` with dependencies
3. Test directory structure: `tests/`
4. Git branch: `001-configuration-manager` (already created ✅)

**Dependencies**:
- `pyyaml>=6.0`
- `pydantic>=2.0`
- `python>=3.10`
- `numpy>=1.20` (for viscosity calculations)

**Acceptance Criteria**:
- [ ] `sister_py/__init__.py` exists and package importable
- [ ] `sister_py/config.py` module importable (initially stub)
- [ ] `pyproject.toml` has all dependencies listed
- [ ] `tests/` directory exists with `test_config.py`
- [ ] Can run `pytest tests/test_config.py` (all tests pending implementation)

**Effort Estimate**: 0.5 days

---

### WP-02: Data Validation Layer (Pydantic BaseModels)

**Objective**: Define all Pydantic validation models with custom validators

**Deliverables**:

1. **11 Pydantic BaseModel classes**:
   ```python
   SimulationConfig, DomainConfig, GridConfig
   DensityParams, DuctileCreepParams
   RheologyConfig, PlasticityParams, ElasticityParams, ThermalParams
   MaterialConfig, BCConfig, PhysicsConfig, SolverConfig
   FullConfig (root)
   ```

2. **Custom validators** for:
   - Positive values (Nt, dt_out, xsize, ysize, A, rho0, etc.)
   - Range constraints (0 < μ < 1, etc.)
   - Monotonicity (grid breaks increasing)
   - Uniqueness (phase IDs unique)
   - Conditional constraints (if plasticity enabled, C and μ required)

3. **Error message formatting**:
   ```python
   @field_validator('mu')
   @classmethod
   def mu_range(cls, v, info):
       if not (0 < v < 1):
           raise ValueError(
               f"friction at {info.field_name} = {v}, "
               f"expected 0 < μ < 1"
           )
       return v
   ```

**Acceptance Criteria**:
- [ ] All 11 models defined and importable
- [ ] Pydantic v2 syntax used (BaseModel, field_validator)
- [ ] Valid config loads without error
- [ ] Invalid mu=1.5 rejected with granular message
- [ ] All custom validators working
- [ ] Error messages include parameter path

**Effort Estimate**: 1.5 days

---

### WP-03: ConfigurationManager & Material Classes

**Objective**: Implement core classes with full functionality

**Deliverables**:

1. **ConfigurationManager class**:
   ```python
   @classmethod
   def load(filepath: str) -> ConfigurationManager
   def __getattr__(name) → nested attribute access
   def get_materials() -> dict[int, Material]
   def to_dict() -> dict
   def to_yaml(filepath: str) -> None
   def to_string() -> str
   def validate() -> None
   ```

2. **Material class**:
   ```python
   def density(T: float) -> float
   def viscosity_ductile(sigma_II, eps_II, T) -> float
   def viscosity_plastic(sigma_II, P) -> float
   def viscosity_effective(sigma_II, eps_II, T, P) -> float
   ```

3. **Viscosity calculations** (SI units):
   - Power-law creep: ε̇ = A·σⁿ·exp(-E/RT) → η = 1/(2·A·σ^(n-1)·exp(-E/RT))
   - Mohr-Coulomb: σ_Y = (C + μ·P)·cos(arctan(μ))
   - Harmonic mean for coupled rheology

4. **Nested attribute access mechanism**:
   - `cfg.DOMAIN.xsize` → getattr(getattr(cfg, 'DOMAIN'), 'xsize')
   - `cfg.MATERIALS[0].density.rho0` → indexing + nested access

**Acceptance Criteria**:
- [ ] ConfigurationManager instantiates without error
- [ ] `load()` classmethod parses YAML
- [ ] Nested attribute access works
- [ ] Material objects created correctly
- [ ] All viscosity methods return correct values
- [ ] Round-trip: load → save → load → identical

**Effort Estimate**: 2 days

---

### WP-04: Example Configurations

**Objective**: Create realistic example YAML files

**Deliverables**:

1. **`sister_py/data/examples/continental_rift.yaml`**
   - Dimensions: 170 km × 60 km
   - Grid: refined zones for weak layer (500 m in center)
   - Materials: sticky layer (phase 1) + mantle (phase 2)
   - Boundary conditions: velocity-driven extension
   - Physics: elasticity + plasticity + thermal

2. **`sister_py/data/examples/subduction.yaml`**
   - Dimensions: 300 km × 150 km
   - Grid: refined slab zone
   - Materials: overriding plate, subducting slab, mantle wedge
   - Boundary conditions: plate convergence
   - Physics: elasticity + plasticity + thermal

3. **`sister_py/data/examples/shear_flow.yaml`**
   - Simple analytical test case
   - Easy to validate numerically

4. **`sister_py/data/defaults.yaml`**
   - Default parameter values for all fields
   - Comments explaining each section
   - Used as reference for users

**Acceptance Criteria**:
- [ ] All 3 example files valid YAML
- [ ] Examples load with ConfigurationManager
- [ ] Examples parse without validation errors
- [ ] Comments explain each section
- [ ] Parameter ranges realistic for geodynamics

**Effort Estimate**: 1 day

---

### WP-05: Test Suite

**Objective**: Comprehensive testing with > 90% coverage

**Deliverables**:

1. **Unit tests** (`tests/test_config.py`):
   - Pydantic validation (valid config loads)
   - Invalid parameters rejected
   - Granular error messages
   - All errors collected
   - Custom validators working

2. **Round-trip tests**:
   - Load YAML → modify dict → save → load → compare
   - 6 sig fig precision maintained
   - Comments preserved

3. **Performance tests**:
   - Config load < 100 ms (1000-line YAML)
   - Material access < 1 μs per viscosity call
   - Benchmark with pytest-benchmark

4. **Integration tests**:
   - Nested attribute access
   - Material object creation
   - Viscosity calculations vs expected values

5. **Edge case tests**:
   - Missing env vars → clear error
   - Invalid file path → FileNotFoundError
   - Overlapping grid zones → validation error
   - All error paths covered

**Acceptance Criteria**:
- [ ] All tests pass (`pytest exit code 0`)
- [ ] Coverage > 90% (`pytest --cov`)
- [ ] Performance benchmarks pass
- [ ] Edge cases handled gracefully
- [ ] Error messages are clear

**Effort Estimate**: 1.5 days

---

### WP-06: Documentation & Examples

**Objective**: Complete API docs and user guides

**Deliverables**:

1. **API Documentation** (in docstrings):
   - ConfigurationManager: class, methods, examples
   - Material: class, methods, equations
   - Pydantic models: all 11 classes documented
   - Error handling: common errors explained

2. **"5-Minute Quick Start" guide**:
   - Copy continental_rift.yaml
   - Modify 3 parameters (xsize, ysize, Nt)
   - Load with ConfigurationManager
   - Access nested values
   - Export to YAML

3. **YAML Schema Documentation**:
   - Each top-level section (SIMULATION, DOMAIN, GRID, MATERIALS, BC, PHYSICS, SOLVER)
   - Parameter descriptions, units, typical ranges
   - Example values

4. **Viscosity Documentation**:
   - Power-law creep equations & parameters
   - Mohr-Coulomb plasticity equations
   - SI units & typical values

**Acceptance Criteria**:
- [ ] All public methods have docstrings
- [ ] Quick-start guide complete
- [ ] YAML schema documented
- [ ] Examples are executable
- [ ] Users can follow guide without errors

**Effort Estimate**: 1 day

---

## 5. Dependencies & Sequencing

### Dependency Graph

```
WP-01 (Setup)
    ↓
WP-02 (Pydantic models)
    ↓
WP-03 (ConfigurationManager & Material) ← depends on WP-02
    ↓
WP-04 (Examples) ← depends on WP-03
    ↓
WP-05 (Tests) ← depends on WP-04
    ↓
WP-06 (Documentation) ← depends on WP-03, WP-04
```

**Critical path**: WP-01 → WP-02 → WP-03 → WP-05 (4 days minimum)

**Parallelizable**: 
- WP-04 and WP-06 can start once WP-03 begins
- Testing can be done incrementally during WP-03

---

## 6. Milestones & Timeline

| Milestone | Target Date | Duration | Status |
|-----------|------------|----------|--------|
| **M1: Setup complete** | Day 0.5 | 0.5 days | Pending |
| **M2: Pydantic models** | Day 2 | 1.5 days | Pending |
| **M3: Core classes** | Day 4 | 2 days | Pending |
| **M4: Examples complete** | Day 5 | 1 day | Pending |
| **M5: Test suite passing** | Day 5.5 | 1.5 days | Pending |
| **M6: Documentation done** | Day 6.5 | 1 day | Pending |
| **M7: Code review** | Day 7 | 0.5 days | Pending |
| **M8: Merge to main** | Day 7 | — | Pending |

**Total estimated effort**: 3-5 days (with parallelization)

---

## 7. Risk Assessment

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|-----------|
| Pydantic v2 API complexity | Medium | Medium | Review official docs, start with simple validators |
| Viscosity formula bugs | Low | High | Compare 10 cases vs MATLAB, unit tests required |
| Round-trip precision issues | Low | High | Use yaml.safe_dump with explicit precision |
| Performance < 100 ms | Low | Low | Profile with pytest-benchmark, optimize if needed |
| Test coverage < 90% | Medium | Medium | Write tests incrementally during implementation |

---

## 8. Constitution Compliance

### 5 Binding Design Principles

All code must satisfy:

| Principle | How ConfigurationManager Complies |
|-----------|-----------------------------------|
| **I. Single-File Input** | One YAML file drives simulation; users modify only parameters |
| **II. Fully-Staggered Grid** | N/A for Phase 0A (used by Grid in Phase 1A) |
| **III. Performance-First** | Config load < 100 ms; no Python objects in data structures |
| **IV. Modular Rheology** | Material class composes ductile + plastic + elastic rheology |
| **V. Test-First** | Coverage > 90%; all acceptance criteria testable |

**Compliance verification**:
- [ ] Single-file paradigm enforced (users can only modify YAML)
- [ ] No Python objects in hot loops (viscosity methods Numba-compatible)
- [ ] Test coverage > 90%
- [ ] All 17 FR + 10 SC binding requirements satisfied

---

## 9. Quality Standards

### Code Quality
- ✅ PEP 8 compliant
- ✅ Type hints throughout
- ✅ Docstrings (Google style)
- ✅ No unused imports
- ✅ No hardcoded values (all configurable)

### Testing Standards
- ✅ Unit tests for each method
- ✅ Integration tests for workflows
- ✅ Performance benchmarks
- ✅ Edge case coverage
- ✅ 90%+ code coverage

### Documentation Standards
- ✅ API docstrings complete
- ✅ Example code executable
- ✅ Error messages helpful
- ✅ YAML schema documented
- ✅ Quick-start guide provided

---

## 10. Acceptance & Handoff

### Acceptance Criteria (Final)

Before code can be merged:

1. ✅ All 17 functional requirements (FR-001 to FR-017) working
2. ✅ All 10 success criteria (SC-001 to SC-010) met
3. ✅ Test coverage > 90%
4. ✅ All tests passing (pytest exit code 0)
5. ✅ Performance benchmarks passing
6. ✅ Constitution compliance verified
7. ✅ Documentation complete
8. ✅ Code review approved
9. ✅ No TODOs or FIXMEs remaining

### Handoff Deliverables

To proceed to Phase 1A (Grid, Material, Markers):

1. ✅ `sister_py/config.py` - Fully implemented
2. ✅ `tests/test_config.py` - > 90% coverage
3. ✅ `sister_py/data/examples/*.yaml` - All examples working
4. ✅ `sister_py/data/defaults.yaml` - Default values provided
5. ✅ `README.md` - Quick-start & schema guide
6. ✅ PR merged to main
7. ✅ v0.1.0-alpha release tagged

---

## 11. Implementation Notes

### Code Structure

```python
# sister_py/config.py

# Pydantic models (11 classes)
class SimulationConfig(BaseModel): ...
class DomainConfig(BaseModel): ...
# ... etc

# Material class
class Material:
    def viscosity_ductile(...): ...
    def viscosity_plastic(...): ...
    def viscosity_effective(...): ...
    def density(T): ...

# ConfigurationManager class
class ConfigurationManager:
    @classmethod
    def load(filepath): ...
    def get_materials(): ...
    def to_yaml(filepath): ...
    def to_dict(): ...
    def to_string(): ...
    def validate(): ...
```

### Key Implementation Details

1. **Environment variable expansion**:
   ```python
   filepath = os.path.expandvars(filepath)  # ${HOME} → /home/user
   ```

2. **Granular error messages**:
   ```python
   raise ValueError(f"friction at {info.field_name} = {v}, expected 0 < μ < 1")
   ```

3. **Viscosity calculations** (SI units):
   ```python
   # Power-law creep
   eta_ductile = 1 / (2 * A * (sigma_II ** (n - 1)) * np.exp(-E / (R * T)))
   
   # Mohr-Coulomb
   sigma_Y = (C + mu * P) * np.cos(np.arctan(mu))
   ```

4. **Round-trip fidelity**:
   ```python
   yaml.dump(data, f, default_flow_style=False, sort_keys=False)
   ```

### Test-First Strategy

**Order of implementation**:
1. Write tests for Pydantic models → implement models
2. Write tests for ConfigurationManager → implement ConfigurationManager
3. Write tests for Material → implement Material
4. Write integration tests → verify everything works together

---

## 12. Glossary

| Term | Definition |
|------|-----------|
| **Pydantic** | Python data validation library (v2) using type hints |
| **BaseModel** | Pydantic class for validating nested data structures |
| **field_validator** | Pydantic decorator for custom validation rules |
| **SI units** | International System of Units (K, Pa, Pa·s, J/mol, m, kg/m³) |
| **Round-trip** | Load → modify → save → load → compare |
| **Granular error** | Error message with parameter path, value, and expected range |
| **Constitution** | 5 binding design principles for entire SiSteR-py project |
| **Harmonic mean** | 1/η_eff = 1/η₁ + 1/η₂ (for coupled viscosity) |

---

## Summary

**Phase 0A ConfigurationManager** is a 3-5 day effort with 6 work packages, clear dependencies, and measurable success criteria.

✅ **Ready to implement** — All risks identified, mitigations planned, Constitution compliance verified.

**Next step**: Begin WP-01 (Project Setup) and proceed through WBS in order.

---

**Branch**: `001-configuration-manager`  
**Specification**: `specs/001-configuration-manager/spec.md`  
**Constitution**: `.specify/memory/constitution.md`  
**Status**: 🟢 READY FOR PLANNING → CODING
