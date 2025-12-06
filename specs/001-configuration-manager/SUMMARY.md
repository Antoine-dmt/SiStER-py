# Phase 0A: ConfigurationManager - Specification Summary

**Branch**: `001-configuration-manager`  
**Status**: Ready for Speckit Implementation  
**Date**: 2025-12-06

## Overview

The ConfigurationManager is the **foundational component** for SiSteR-py. It:
- Loads YAML configuration files with full validation
- Provides nested attribute access to all parameters
- Creates Material objects for rheology computations
- Exports configs for reproducibility (round-trip fidelity)

## Key Features

### ✅ Validation (Pydantic v2)
- Granular error messages with parameter paths
- All errors collected (not just first)
- Custom validators for range checks, monotonicity, uniqueness
- SI units enforced (K, Pa, Pa·s, J/mol, m, kg/m³)

### ✅ Nested Attribute Access
```python
cfg.DOMAIN.xsize              # 170000.0
cfg.MATERIALS[0].density.rho0 # 1000.0
cfg.BC['top'].vx              # 1e-10
```

### ✅ Material Objects
```python
materials = cfg.get_materials()
eta = materials[1].viscosity_ductile(sigma_II=1e7, eps_II=1e-15, T=1200)
```

### ✅ Export Methods
- `to_yaml(filepath)` – Save with comments, 6 sig figs
- `to_dict()` – JSON-serializable nested dict
- `to_string()` – Formatted text for stdout/logging
- `validate()` – Re-validate after programmatic changes

## User Stories (Priority-Ordered)

| Priority | Story | Independent Test |
|----------|-------|------------------|
| P1 | Load & validate YAML | Valid config loads without error |
| P1 | Access parameters | Nested attribute access works |
| P1 | Create Material objects | `get_materials()` returns working dict |
| P2 | Export config | Round-trip: load → modify → save → load → identical |
| P3 | String representation | `to_string()` and `to_dict()` work |

## Success Criteria (Measurable)

- Config load: < 100 ms for 1000-line YAML
- Error granularity: all 3+ errors collected and shown
- Round-trip fidelity: bit-identical after load → modify → save → load
- Viscosity accuracy: matches MATLAB SiSteR to 6 sig figs
- Test coverage: > 90% for config.py
- User accessibility: copy example YAML, modify 3-4 params, run

## Example Usage

```python
from sister_py.config import ConfigurationManager

# Load and validate
cfg = ConfigurationManager.load("continental_rift.yaml")

# Access parameters
print(cfg.DOMAIN.xsize)  # 170000.0

# Create materials
materials = cfg.get_materials()
eta = materials[1].viscosity_ductile(1e7, 1e-15, 1200)

# Modify and re-validate
cfg.SIMULATION.Nt = 100
cfg.validate()

# Export for reproducibility
cfg.to_yaml("my_run.yaml")
```

## Deliverables

1. **sister_py/config.py** – ConfigurationManager, Material, Pydantic models
2. **tests/test_config.py** – Unit, round-trip, performance, integration tests
3. **sister_py/data/examples/*.yaml** – continental_rift.yaml, subduction.yaml, shear_flow.yaml
4. **sister_py/data/defaults.yaml** – Sensible defaults
5. **Documentation** – Docstrings, "5-Minute Quick Start", schema guide

## Specification Document

Full specification with all requirements, constraints, and acceptance criteria:
→ `specs/001-configuration-manager/spec.md`

Detailed Speckit prompt with implementation guidelines:
→ `.specify/prompts/phase-0a-configuration-manager.md`

Project Constitution (binding design principles):
→ `.specify/memory/constitution.md`

## Next Steps

1. ✅ **Spec created** (this document)
2. 📋 **Ready for Speckit submission** – Use `.specify/prompts/phase-0a-configuration-manager.md`
3. 🚀 **Development** – Speckit agent implements ConfigurationManager end-to-end
4. ✅ **Testing** – All acceptance criteria verified
5. 🔀 **PR & Review** – Code review, Constitution compliance check
6. 📦 **Phase 1A** – Continue with FullyStaggeredGrid (depends on ConfigurationManager)

---

**Ready to submit to Speckit when you are!** 🎯
