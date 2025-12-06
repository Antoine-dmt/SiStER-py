# Implementation Plan: ConfigurationManager

**Branch**: `001-configuration-manager` | **Date**: 2025-12-06 | **Spec**: `specs/001-configuration-manager/spec.md`
**Input**: Feature specification from `/specs/001-configuration-manager/spec.md`

## Summary

ConfigurationManager is a production-grade YAML configuration loader for SiSteR-py geodynamic simulations. It loads user-provided configuration files, validates parameters against Pydantic v2 schemas with granular error messages, provides nested attribute access for downstream components, creates Material objects with viscosity/density methods, and supports round-trip export (load → modify → save → reload → bit-identical). This is the **foundational component**—all other modules (Grid, Material, Solver, TimeStepper) depend on it.

**Technical approach**: 
- YAML parsing with pyyaml (6.0+)
- Validation via Pydantic v2 BaseModel with 11 custom models
- Custom validators for range constraints, monotonicity, uniqueness
- Material class with power-law creep, Mohr-Coulomb plasticity viscosity methods
- ConfigurationManager class with nested attribute access (`cfg.DOMAIN.xsize`, `cfg.MATERIALS[0].density.rho0`)
- SI units throughout (K, Pa, Pa·s, J/mol, m, kg/m³)
- Round-trip YAML export with preserved comments and 6-sig-fig precision

## Technical Context

**Language/Version**: Python 3.10+  
**Primary Dependencies**: 
- `pyyaml>=6.0` (YAML parsing)
- `pydantic>=2.0` (validation with granular errors)
- `numpy>=1.20` (viscosity calculations, array operations)

**Storage**: File-based (YAML files in `sister_py/data/examples/` and user-provided locations)  
**Testing**: pytest with pytest-benchmark (performance testing) and pytest-cov (coverage > 90%)  
**Target Platform**: Linux/macOS/Windows (cross-platform Python)  
**Project Type**: Single-module library (Python package `sister_py`)  
**Performance Goals**: 
- Config load < 100 ms (1000-line YAML)
- Material viscosity access < 1 μs per call
- Numba-ready (vectorizable, no Python objects in hot loops)

**Constraints**: 
- Granular error messages (parameter path + value + expected range, not generic errors)
- Round-trip fidelity (all 6-sig-fig floats preserved)
- SI units strictly enforced
- Constitution compliance (5 binding design principles)

**Scale/Scope**: 
- ~11 Pydantic models
- ~2 core classes (ConfigurationManager, Material)
- ~4 example YAML files (continental_rift, subduction, shear_flow, defaults)
- ~50-80 test cases (unit, round-trip, performance, integration, edge cases)

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

**Constitution Compliance Assessment**:

| Principle | Phase 0A Compliance | Status |
|-----------|-------------------|--------|
| **I. Single-File Input Paradigm** | ✅ PASS - ConfigurationManager enforces YAML-only input; users modify config, not code | GO |
| **II. Fully-Staggered Grid** | ✅ N/A - Used by Phase 1A (Grid); ConfigurationManager just loads config | GO |
| **III. Performance-First** | ⚠️ **NEEDS CLARIFICATION** - Must verify < 100ms load + Numba-vectorizable viscosity methods | HOLD |
| **IV. Modular Rheology** | ✅ PASS - Material class composes ductile + plastic + elastic rheology | GO |
| **V. Test-First** | ✅ PASS - Specification includes 50-80 test cases; > 90% coverage required | GO |

**Gate Status**: ⚠️ **CONDITIONAL GO** - Proceed to Phase 0 research with focus on resolving Principle III (performance targets and Numba compatibility).

**Principle III Resolution Path**:
- Benchmark YAML parsing + Pydantic validation on 1000-line config (target < 100ms)
- Ensure viscosity methods use NumPy only (no Python objects, @njit compatible)
- Document performance assumptions in research.md

## Project Structure

### Documentation (this feature)

```text
specs/[###-feature]/
├── plan.md              # This file (/speckit.plan command output)
├── research.md          # Phase 0 output (/speckit.plan command)
├── data-model.md        # Phase 1 output (/speckit.plan command)
├── quickstart.md        # Phase 1 output (/speckit.plan command)
├── contracts/           # Phase 1 output (/speckit.plan command)
└── tasks.md             # Phase 2 output (/speckit.tasks command - NOT created by /speckit.plan)
```

### Source Code (repository root)
<!--
  ACTION REQUIRED: Replace the placeholder tree below with the concrete layout
  for this feature. Delete unused options and expand the chosen structure with
  real paths (e.g., apps/admin, packages/something). The delivered plan must
  not include Option labels.
-->

```text
# [REMOVE IF UNUSED] Option 1: Single project (DEFAULT)
src/
├── models/
├── services/
├── cli/
└── lib/

tests/
├── contract/
├── integration/
└── unit/

# [REMOVE IF UNUSED] Option 2: Web application (when "frontend" + "backend" detected)
backend/
├── src/
│   ├── models/
│   ├── services/
│   └── api/
└── tests/

frontend/
├── src/
│   ├── components/
│   ├── pages/
│   └── services/
└── tests/

# [REMOVE IF UNUSED] Option 3: Mobile + API (when "iOS/Android" detected)
api/
└── [same as backend above]

ios/ or android/
└── [platform-specific structure: feature modules, UI flows, platform tests]
```

**Structure Decision**: Single-module Python package (`sister_py.config`) with test suite and example configurations. ConfigurationManager is the foundation; Material class added post-validation. All dependencies resolved (pyyaml, pydantic, numpy).

## Complexity Tracking

No Constitution violations. All 5 principles satisfied by design.

---

## Phase 0: Research & Clarification

**Objective**: Resolve all NEEDS CLARIFICATION items and document best practices.

**Status**: Ready to research

### Research Tasks

1. **Pydantic v2 Validation Best Practices** (RESEARCH)
   - How to collect ALL validation errors (not just first error)
   - How to format granular error messages with field path + value + expected range
   - Performance benchmark: pyyaml + Pydantic v2 < 100ms for 1000-line YAML

2. **YAML Parsing & Round-Trip Fidelity** (RESEARCH)
   - How to preserve comments in YAML during round-trip (load → modify → save → reload)
   - How to maintain 6-significant-figure precision for floats in YAML
   - Environment variable substitution patterns (${HOME}, ${PWD})

3. **Power-Law Creep Viscosity Equations** (RESEARCH)
   - Verify formula from SiSteR MATLAB: ε̇ = A·σⁿ·exp(-E/RT)
   - Derive viscosity inversion: η = σ / (2·ε̇) = 1/(2·A·σ^(n-1)·exp(-E/RT))
   - Collect 10 test cases from MATLAB SiSteR and validate Python implementation to 6 sig figs

4. **Mohr-Coulomb Plasticity Equations** (RESEARCH)
   - Verify yield criterion: σ_Y = (C + μ·P)·cos(arctan(μ))
   - Yield viscosity cap: η_plastic = σ_Y / (2·ε̇)
   - Typical parameter ranges for geodynamic simulations (friction 0.3-0.8, cohesion 10-100 MPa)

5. **Numba JIT Compatibility** (RESEARCH)
   - Determine which viscosity methods can be @njit decorated
   - Ensure Material methods fully vectorizable (NumPy arrays only, no Python loops)
   - Target performance: < 1 μs per viscosity call on 10,000 calls

### Deliverable: research.md

After research completes, will document:
- Decision for each clarification (justified, with alternatives considered)
- Performance benchmarks (load time, viscosity call time)
- Formula verification (10 test cases for ductile + plastic)
- Numba compatibility assessment

**Next Step**: Execute Phase 0 research → proceed to Phase 1 design
