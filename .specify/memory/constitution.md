# SiSteR-py Constitution
A production-grade geodynamic simulation framework combining fully-staggered grid accuracy, performance optimization, and accessible YAML-based configuration.

## Core Principles

### I. Single-File Input Paradigm
Configuration-driven simulation: one YAML file drives entire execution (mirroring SiSteR MATLAB design). Users modify only config parameters; code unchanged. All simulations must be reproducible via saved config files. YAML format (not JSON) ensures human readability, comments, and version control compatibility.

### II. Fully-Staggered Grid for Accuracy
Grid method must follow Duretz, May, Gerya (2013) fully-staggered discretization: pressure at cell centers, velocities at maximally-separated face centers (checkerboard pattern). This reduces discretization error by 30-50% vs standard staggered grids, critical for variable viscosity simulations (10¹⁸ to 10²⁵ Pa·s). 7-point compact FD stencil maintains sparsity and solver efficiency.

### III. Performance-First (Numba-Ready Code)
Every hot loop (grid creation, interpolation, matrix assembly) marked @njit or @vectorize for Numba JIT compilation. Target: grid creation < 10ms (1000×1000 cells), matrix assembly < 100ms. Use NumPy broadcasting exclusively; no nested Python objects in loops. Defer GPU acceleration (CuPy) to Phase 5. Benchmark & profile mandatory before release.

### IV. Modular Rheology System
Physics decoupled into composable modules: DuctileRheology, PlasticRheology, ElasticRheology. Each Material object composes rheology instances; viscosity coupling handled explicitly. New physics models added without modifying existing code. All stress/strain computations vectorized (2D arrays, not element-by-element).

### V. Test-First Implementation (NON-NEGOTIABLE)
Unit tests written → acceptance criteria validated → test fails → implementation. Acceptance criteria from prompts are binding. Round-trip testing mandatory (load YAML → run → save → reload → verify identity). Analytical solutions provided for grid/solver validation. Coverage target: > 90% for solver, > 80% overall.

## Configuration & Accessibility Standards

**YAML Schema**: Must match SiSteR MATLAB input structure (SIMULATION, DOMAIN, GRID, MATERIALS, BC, PHYSICS, SOLVER sections). Validation via Pydantic v2 with granular error messages ("friction at MATERIALS[1].mu = 1.5, expected 0 < μ < 1", not generic "validation failed"). Comments preserved in YAML files. Environment variable substitution supported (${HOME}/data/).

**Material Properties**: SI units throughout (Pa, Pa·s, K, J/mol, m, kg/m³). Temperature in Kelvin, not Celsius. Pre-exponential factor A in [Pa^(-n)·s^(-1)]. Viscosity bounds enforced: 10^18 to 10^25 Pa·s. Friction coefficients: 0 < μ < 1. All parameters with sensible defaults; users override what they need.

**Accessibility Requirements**: Example YAML configs in `sister_py/data/examples/`. Docstrings include typical parameter values (e.g., "mid-lithosphere viscosity: 1e20-1e21 Pa·s"). Error messages suggest valid ranges. Installation: `pip install sister-py` includes examples in ~/.sister_py/. Quick-start guide: load example YAML, modify 3-4 parameters, run. Performance: config load < 100 ms.

## Integration & Phase Dependencies

**Execution Order (Strict)**:
- Phase 0: Config system (Prompt 0A) → foundational
- Phase 1: Grid (Prompt 1A) + Material (Prompt 2A) + Markers (Prompt 3A) → parallel, each independent
- Phase 2: Assembly (Prompt 4A) + Solver (Prompt 5A) → sequential (assembly feeds solver)
- Phase 3: TimeStepper (Prompt 6A) → orchestrates all Phase 1-2
- Phase 4: Distribution (package structure, CI/CD) → last
- Phase 5: Optimization (Numba, profiling) → optional

**Component Interfaces**: ConfigurationManager → FullyStaggaredGrid, Material, MarkerSwarm (all receive config objects). All components export to_dict() for serialization. Grid provides index_* and coord_* methods for assembly. Solver receives (matrix, RHS, viscosity) → solution vector.

## Governance

**Amendment Process**: Constitution changes require explicit justification (error in physics, performance bottleneck, accessibility failure). Amendments trigger version bump (MAJOR for principle removal, MINOR for clarification, PATCH for wording). All PRs reference which principles they satisfy. Code review checklist: (1) Principle compliance, (2) Acceptance criteria met, (3) Tests pass, (4) Docstrings updated.

**Compliance Verification**: 
- Grid creation benchmarked vs 10ms target
- Interpolation test: constant field → verify exact, linear field → verify to machine precision
- Config round-trip: load → modify → save → reload → byte-identical
- Solver convergence: Picard iterations < 100, Newton superlinear
- All @njit methods compile without error

**Runtime Guidance**: See `.specify/guidance/speckit-phase-workflow.md` for Speckit submission order and prompt customization. See `SPECKIT_PROMPTS_ENHANCED.md` for full prompt templates.

**Version**: 1.0.0 | **Ratified**: 2025-12-06 | **Last Amended**: 2025-12-06
