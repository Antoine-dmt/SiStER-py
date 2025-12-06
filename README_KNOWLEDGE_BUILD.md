# SiSteR-py Project: Knowledge Build Complete ✓

## Executive Summary

You now have comprehensive context for redesigning SiSteR (MATLAB → Python/OOP):

### Three Knowledge Documents Created

1. **SISTER_KNOWLEDGE_CONTEXT.md** (Main Overview)
   - What is SiSteR? Purpose, capabilities, characteristics
   - Stokes equation fundamentals & how they're solved
   - Complete time-stepping algorithm
   - Computational strategy (staggered grid, Picard/Newton iteration)
   - Rheology models (ductile, plastic, elastic)
   - Marker-based tracking system
   - Continental rifting example
   - OOP redesign opportunities

2. **STOKES_MATHEMATICS.md** (Technical Deep Dive)
   - Continuous equations in component form
   - Staggered grid discretization & layout
   - Complete finite difference stencils
   - System matrix assembly pseudocode
   - Boundary condition implementations
   - Non-linear viscosity coupling
   - Numerical stability & scaling
   - Analytical validation examples
   - Implementation checklist

3. **SPECKIT_PROMPT_DESIGN.md** (Next Steps)
   - Speckit fundamentals for this project
   - Prompt template structure
   - Concrete prompt examples:
     - Phase 1: StokesGrid, Material, Marker classes
     - Phase 2: Interpolation & MarkerSwarm
     - Phase 3: Matrix assembly, non-linear solver
     - Phase 4: Time stepper & main loop
   - Best practices & checklist
   - Recommended execution order

---

## Key Insights from Knowledge Building

### 1. What SiSteR Does

- Simulates **geological deformation** over millions of years
- Solves the **Stokes equations** (slow viscous flow) with **non-linear rheology**
- Uses **Eulerian grid** (fixed in space) for solving physics
- Tracks **Lagrangian markers** (moving particles) for material history
- Couples: momentum + continuity + rheology + advection + thermal diffusion

### 2. The Central Algorithm

```
TIME LOOP:
  1. Interpolate marker properties → grid nodes
  2. SOLVE STOKES (non-linear):
     - Picard iterations: viscosity(ε̇) → solve LS=R
     - Newton iterations: faster convergence near solution
     - ~10-100 iterations per time step, converge to tol ~1e-9
  3. Interpolate strain rate → markers
  4. Update marker stresses (VEP coupling)
  5. OUTPUT (save v, p, T, viscosity, markers every N steps)
  6. Advect markers (Lagrangian advection)
  7. Reseed markers (maintain density)
  8. Advance time (CFL-limited time step)
```

**Why separate Eulerian + Lagrangian?**
- Grid: accurate PDE solving, fast
- Markers: track material history, enable large deformations without remeshing

### 3. The Physics: Why Non-Linear?

Viscosity depends on **stress** or **strain rate**:

$$\eta = \eta(\dot{\varepsilon}_{II}, \sigma_{II}, T)$$

- **Hotter** → lower viscosity (thermal activation)
- **Higher strain rate** → lower viscosity (power-law creep)
- **Higher stress** → plastic yield if too large (brittle failure)

This **non-linearity** requires iteration: can't just solve once.

### 4. The Numerics: Why Staggered Grid?

**Collocated grid** (all variables at same points):
- ❌ Oscillatory solutions (pressure checkerboard)
- ❌ Stability issues

**Staggered grid** (pressure at corners, velocity at edges):
- ✓ Stable, smooth solutions
- ✓ Natural discretization of divergence-free conditions
- ✓ Used in most production codes (Fluent, OpenFOAM, etc.)

### 5. The Challenges

| Challenge | Reason | Solution |
|-----------|--------|----------|
| **Ill-conditioned matrix** | Viscosity ranges 10^18 - 10^25 Pa·s | Scale equations before assembly |
| **Non-linearity** | Viscosity depends on solution | Iterate: Picard (robust) → Newton (fast) |
| **Coupled physics** | Stress, strain rate, temperature all interact | Operator splitting or fully coupled |
| **Advection errors** | Interpolation → diffusion | Use high-order schemes, fine mesh |
| **Localization** | Plastic zones narrow to few cells | Adaptive refinement |

---

## OOP Design Philosophy

### Current MATLAB Limitations

1. **No Encapsulation**: Grid info scattered across functions
2. **Implicit Dependencies**: Variable names magical (Nx, Ny, dx, dy global)
3. **Hard to Extend**: Add new rheology? → modify 5 files
4. **Poor Testability**: No way to test `SiStER_assemble_L_R` in isolation
5. **Performance**: Interpreted, dense matrices, limited parallelization

### Proposed Python/OOP Structure

```python
# Core Data Structures
class StokesGrid:
    """Coordinates, indexing, interpolation"""

class Material(ABC):
    """Base: viscosity(state) → returns η"""
    
class DuctileRheology(Material):
    """Power-law creep"""
    
class PlasticRheology(Material):
    """Mohr-Coulomb yield"""
    
class Phase:
    """Phase ID + material properties"""

class Marker:
    """Single particle: position, phase, stress, strain history"""

class MarkerSwarm:
    """Collection: advect, reseed, interpolate"""

# Solvers
class StokesMatrixAssembler:
    """Builds L and R from grid, viscosity, BC"""

class StokesNonlinearSolver:
    """Picard + Newton iteration until convergence"""

class StokesFlow:
    """Orchestrates: assemble → solve → extract"""

# Integration
class GeodynamicsSimulation:
    """Main loop: step → advect → update → output"""
```

**Benefits**:
- ✓ Clear responsibility per class
- ✓ Testable: each class has unit tests
- ✓ Extensible: add new rheology = subclass Material
- ✓ Reusable: `StokesFlow` can be used standalone
- ✓ Performant: NumPy vectorization + Numba JIT + GPU ready

---

## Recommended Execution Path

### Phase 0: Setup (This Week)
- [ ] Create project skeleton (git repo, pyproject.toml, tests/ folder)
- [ ] Set up CI/CD (pytest, coverage)
- [ ] Decide on performance targets (grid size, time to solution)

### Phase 1: Core Data Structures (1-2 weeks)
Use Speckit for:
1. **StokesGrid** → Create grid, interpolation, indexing
2. **Material & Phase** → Properties, rheology methods
3. **Marker & MarkerSwarm** → Particle tracking, batch operations

**Validation**: Write tests that compare to MATLAB outputs

### Phase 2: Solver Core (2-3 weeks)
Use Speckit for:
4. **StokesMatrixAssembler** → Build FD system
5. **StokesNonlinearSolver** → Picard/Newton iteration
6. **StokesFlow** → Orchestrate solve

**Validation**: Test on analytical solutions (shear flow, gravity column)

### Phase 3: Time Integration (1-2 weeks)
Use Speckit for:
7. **GeodynamicsSimulation** → Main loop
8. **I/O & Checkpointing** → Save/load states

**Validation**: Run continental rifting example, compare to MATLAB

### Phase 4: Optimization (2-3 weeks)
- Profile code: where is time spent?
- Numba-JIT hot loops (interpolation, strain rate calculation)
- Sparse linear algebra tricks (preconditioners, iterative solvers)
- Optional: GPU acceleration with CuPy

### Phase 5: Advanced Features (Ongoing)
- 3D extension
- AMR (adaptive mesh refinement)
- MPI parallelization
- Modern rheology (damage, strain-rate-induced anisotropy)

---

## Speckit Workflow

### When You're Ready (Likely Next Meeting)

1. **Pick ONE component** (e.g., StokesGrid)
2. **Refine the prompt** from SPECKIT_PROMPT_DESIGN.md with specifics from MATLAB code
   - Look up exact index orderings from `SiStER_Initialize`
   - Copy parameter ranges from continental rifting input file
   - Reference exact MATLAB variable names
3. **Submit to Speckit** with:
   - Context (what this fits into)
   - Specification (what to build)
   - Requirements (functional + non-functional)
   - Constraints (design patterns, dependencies)
   - Acceptance criteria (3-5 tests)
   - Example usage (show API)
4. **Review generated code**, iterate if needed
5. **Move to next component**

### Key Prompting Tips

- **Be specific**: "Staggered grid, 2D, variable spacing" not "create a grid"
- **Include numbers**: "10 iterations < 1 second" not "should be fast"
- **Reference original**: "Match MATLAB index ordering from SiStER_Initialize.m line 12"
- **Provide examples**: Show how next component will call this
- **Define tests**: "Test with analytical solution: linear velocity field should be exact"

---

## Questions to Explore Before First Speckit Prompt

1. **Target Grid Size**: What's your typical mesh?
   - MATLAB example: 100×50 (170 km × 60 km domain)
   - Acceptable range: 50×30 (small, fast) to 500×300 (realistic, slower)

2. **Time Steps**: How many iterations typical?
   - MATLAB example: Nt=1600 (1600 time steps)
   - Typical sim: 10-100 Myr = 1000-10000 time steps

3. **Solver Choice**: Direct or iterative?
   - Small grids (< 150×150): direct sparse (scipy.sparse.linalg.spsolve)
   - Large grids: iterative (scipy.sparse.linalg.gmres + preconditioner)

4. **Performance Target**: How fast do you need?
   - For research: 10-100 time steps/hour
   - For exploration: 1000+ time steps/hour (requires optimization)

5. **Feature Priority**: What matters most?
   - Correctness vs. MATLAB? (need regression tests)
   - New physics (3D, damage)? (affects class design)
   - GPU acceleration? (affects array library choice: NumPy vs. CuPy)
   - Publication readiness? (affects code quality, docs)

---

## File Locations

All knowledge docs are in: `c:\Users\AntoineDemont\Desktop\Perso\Git_projects\SiSteR-py\`

1. **SISTER_KNOWLEDGE_CONTEXT.md** — Main reference
2. **STOKES_MATHEMATICS.md** — Technical equations & stencils
3. **SPECKIT_PROMPT_DESIGN.md** — Design methodology & examples

**MATLAB Source**: Same directory, `SiStER-master/` folder

---

## Next Actions

### Immediately
- [ ] Read through the three knowledge documents (order: 1 → 3 → 2)
- [ ] Open MATLAB files, cross-reference against knowledge docs
- [ ] Identify any gaps in your understanding
- [ ] Ask clarifying questions

### This Week
- [ ] Set up Python project structure
- [ ] Design your target architecture (draw class diagrams)
- [ ] Decide: NumPy-only vs. Numba vs. GPU?

### Next Meeting
- [ ] Refine first Speckit prompt (StokesGrid or Material)
- [ ] Prepare test cases from MATLAB
- [ ] Start implementation

---

## Success Criteria

By the end of Phase 1 (core data structures):
- ✓ Can create a grid matching MATLAB
- ✓ Can initialize markers on that grid
- ✓ Can interpolate a test field from nodes to markers and back
- ✓ Numerical tests pass: compare to MATLAB, errors < 1e-6

By end of Phase 2 (solver):
- ✓ Can assemble Stokes matrix for simple test case
- ✓ Solver converges in < 50 iterations on continental rifting
- ✓ Solution matches MATLAB first time step to 4 significant figures

By end of Phase 3 (time integration):
- ✓ Run 100 time steps of continental rifting
- ✓ Save/load checkpoints correctly
- ✓ Marker count stable after reseeding

By end of Phase 4 (optimization):
- ✓ 1000 time steps in < 1 hour (or your target)
- ✓ Profile shows main bottleneck identified
- ✓ Code clean, well-tested, documented

---

## Resources

### Theory
- Gerya & Yuen (2003) "Characteristics-based marker-in-cell method" — Foundational
- Elman, Silvester, Wathen (2014) "Finite Elements and Fast Iterative Solvers" — Stokes solvers
- Turcotte & Schubert (2014) "Geodynamics" — Physics background

### Numerical Methods
- SciPy sparse.linalg documentation
- PETSc/Firedrake (if you decide on MPI later)
- Numpy performance tips

### Existing Codes (for ideas)
- **Underworld** (Python, geodynamics, does similar things)
- **ASPECT** (C++, large-scale, modern design)
- **code_aster** (FEM, open source)

---

## You're Ready!

You now understand:
1. **What** SiSteR does (geodynamic simulation)
2. **How** it works (Stokes equations on staggered grid, non-linear iteration, marker advection)
3. **Why** each design choice (stability, accuracy, extensibility)
4. **When** to use Speckit (scaffolding modular components)
5. **How** to prompt Speckit (concrete, specific, testable)

**The knowledge base is solid. Implementation can proceed confidently.**

Questions? Review the three documents or ask — happy to clarify any concept!

🚀 **Go build something awesome!**
