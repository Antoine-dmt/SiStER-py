# Speckit Phase 0A: ConfigurationManager

## 🎯 Quick Start

This folder contains the **specification and implementation guidance** for Phase 0A of SiSteR-py: ConfigurationManager.

### Files in This Directory

1. **phase-0a-configuration-manager.md** ← **START HERE**
   - Full Speckit prompt with all requirements, constraints, acceptance criteria
   - Ready to submit to Speckit coding agent
   - Contains example YAML, test strategy, dependencies

2. **.specify/memory/constitution.md**
   - Project Constitution with 5 binding design principles
   - Governs all implementations across all phases
   - Speckit must comply with Constitution

### Implementation Status

| Step | Status | Details |
|------|--------|---------|
| Branch Creation | ✅ Complete | `001-configuration-manager` created and checked out |
| Specification | ✅ Complete | `specs/001-configuration-manager/spec.md` (161 lines) |
| Constitution | ✅ Complete | `.specify/memory/constitution.md` (SiSteR-py design principles) |
| Speckit Prompt | ✅ Complete | `.specify/prompts/phase-0a-configuration-manager.md` (ready to submit) |
| Implementation | ⏳ Ready | Waiting for Speckit agent to implement |
| Testing | ⏳ Ready | Test strategy defined in spec |
| Code Review | ⏳ Pending | Will verify Constitution compliance |

## 📋 What Speckit Will Build

The ConfigurationManager component consists of:

```
sister_py/config.py
├── ConfigurationManager class
│   ├── load(filepath) → ConfigurationManager
│   ├── get_materials() → dict[int, Material]
│   ├── to_yaml(filepath) → None
│   ├── to_dict() → dict
│   ├── to_string() → str
│   └── validate() → None
│
├── Material class
│   ├── viscosity_ductile(sigma_II, eps_II, T) → float
│   ├── viscosity_plastic(sigma_II, P) → float
│   ├── viscosity_effective(...) → float
│   └── density(T) → float
│
└── Pydantic v2 BaseModel classes
    ├── SimulationConfig
    ├── DomainConfig
    ├── GridConfig
    ├── MaterialConfig
    ├── BCConfig
    ├── PhysicsConfig
    ├── SolverConfig
    └── FullConfig
```

Plus:
- `tests/test_config.py` – Unit, round-trip, performance, integration tests
- `sister_py/data/examples/*.yaml` – Example configurations
- `sister_py/data/defaults.yaml` – Default parameter values

## 🚀 How to Submit to Speckit

1. **Review the Speckit Prompt**
   ```bash
   cat .specify/prompts/phase-0a-configuration-manager.md
   ```

2. **Customize if Needed** (optional)
   - Adjust acceptance criteria if you have domain-specific requirements
   - Add YAML examples specific to your use cases
   - Modify test strategy if needed

3. **Submit to Speckit**
   - Use `/speckit.specify` mode with the prompt content
   - Or reference the file path directly in your Speckit system

4. **Monitor Implementation**
   - Speckit will create feature branch (already exists: `001-configuration-manager`)
   - Development happens on this branch
   - Pull request created automatically when complete

## 📐 Design Principles (from Constitution)

All implementations must satisfy these **binding principles**:

1. **Single-File Input Paradigm** → One YAML drives entire simulation
2. **Fully-Staggered Grid for Accuracy** → Not applicable to ConfigurationManager (Phase 1A concern)
3. **Performance-First (Numba-Ready)** → Config load < 100 ms, no Python objects in data
4. **Modular Rheology System** → Material objects compose rheology models
5. **Test-First Implementation** → Tests written before code, coverage > 90%

## ✅ Acceptance Criteria (What Speckit Must Deliver)

- [ ] Load `continental_rift.yaml` without errors (real MATLAB input converted)
- [ ] Reject invalid config with **granular** error messages (not generic "validation failed")
- [ ] Round-trip: load → modify → save → load → bit-identical
- [ ] Performance: load 1000-line config in < 100 ms
- [ ] Export: `config.to_yaml(file)` maintains 6 significant figures
- [ ] Materials: `config.get_materials()` returns dict of Material objects
- [ ] All validators working (custom checks, range validation, etc.)
- [ ] Comments preserved after round-trip
- [ ] Environment variables expanded: `${HOME}/data/` → actual path
- [ ] Nested access works: `cfg.DOMAIN.xsize`, `cfg.MATERIALS[0].density.rho0`

## 🔗 Related Files

**Knowledge Base** (context for implementation):
- `SISTER_KNOWLEDGE_CONTEXT.md` – SiSteR MATLAB overview & algorithm
- `STOKES_MATHEMATICS.md` – Mathematical background on Stokes equations
- `ARCHITECTURE_VISUAL_GUIDE.md` – System design and workflows

**Project Structure**:
- `SiStER-master/` – Original MATLAB SiSteR code (reference)
- `.specify/` – Speckit framework files
- `specs/001-configuration-manager/` – This feature's specification
- `.specify/memory/constitution.md` – Project Constitution (binding)
- `.specify/prompts/phase-0a-configuration-manager.md` – Speckit prompt

## 🎓 Example: What a User Will Do

```python
from sister_py.config import ConfigurationManager

# 1. Copy example YAML from package
cfg = ConfigurationManager.load("~/.sister_py/examples/continental_rift.yaml")

# 2. Modify a few parameters programmatically or in YAML
cfg.SIMULATION.Nt = 100  # Override time steps
cfg.SOLVER.conv_tol = 1e-8

# 3. Re-validate after changes
cfg.validate()

# 4. Create Material objects for use in solver
materials = cfg.get_materials()
grid_config = cfg.GRID  # Pass to Grid initialization
bc_config = cfg.BC      # Pass to Solver boundary conditions

# 5. Run simulation (Phase 3A, TimeStepper will use ConfigurationManager)
sim = GeodynamicsSimulation(cfg)
sim.run(output_dir="results/")

# 6. Config auto-saved with outputs
cfg.to_yaml("results/config.yaml")  # Reproducibility
```

## 📞 Questions?

If implementation details need clarification:
1. Check `specs/001-configuration-manager/spec.md` for full requirements
2. Review `.specify/prompts/phase-0a-configuration-manager.md` for implementation guidance
3. Refer to Constitution (`.specify/memory/constitution.md`) for design principles
4. Check MATLAB SiSteR examples in `SiStER-master/` for domain context

---

**Status**: ✅ Ready for Speckit Implementation  
**Branch**: `001-configuration-manager`  
**Created**: 2025-12-06  
**Next Phase**: Phase 1A - FullyStaggeredGrid (depends on ConfigurationManager)
