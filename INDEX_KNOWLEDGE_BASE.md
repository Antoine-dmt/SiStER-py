# SiSteR-py Knowledge Base: Complete Index

## 📚 Documentation Overview

You have been provided with a **complete knowledge base** for redesigning SiSteR from MATLAB to Python/OOP with performance optimization. This document serves as the **master index**.

---

## 📖 Core Documents (Read in This Order)

### 1. **README_KNOWLEDGE_BUILD.md** ← START HERE
   - **Purpose**: Executive summary and next steps
   - **Read Time**: 20 minutes
   - **Contains**:
     - What SiSteR does (brief)
     - Key insights from knowledge building
     - Recommended execution path (phases 0-5)
     - Questions to explore before starting
     - Success criteria for each phase
   - **Action**: Read this first to understand the big picture

---

### 2. **SISTER_KNOWLEDGE_CONTEXT.md** ← MAIN REFERENCE
   - **Purpose**: Comprehensive overview of SiSteR code and concepts
   - **Read Time**: 45 minutes (or reference as needed)
   - **Contains** (9 sections):
     1. SiSteR Overview (purpose, characteristics)
     2. Stokes Equation Fundamentals
     3. SiSteR Computational Strategy
     4. Rheology Models (ductile, plastic, elastic)
     5. Marker-Based Tracking
     6. Continental Rifting Example (default test case)
     7. Key Computational Challenges
     8. Python/OOP Redesign Opportunities (class structure)
     9. Key Input Parameters (reference table)
   - **Action**: Keep as primary reference while coding

---

### 3. **STOKES_MATHEMATICS.md** ← TECHNICAL DETAILS
   - **Purpose**: Mathematical foundations and numerical implementation
   - **Read Time**: 60 minutes (or reference as needed)
   - **Contains** (10 sections):
     1. Continuous Stokes equations (PDEs)
     2. Staggered grid discretization
     3. Finite difference stencils
     4. Complete system matrix assembly
     5. Boundary conditions (Dirichlet, Neumann, pressure anchor)
     6. Non-linear rheology coupling
     7. Numerical stability & scaling (Kc, Kb)
     8. Validation via analytical solutions
     9. Code implementation checklist
     10. Key equations summary
   - **Action**: Reference when implementing solvers

---

### 4. **ARCHITECTURE_VISUAL_GUIDE.md** ← DESIGN REFERENCE
   - **Purpose**: Visual and conceptual architecture
   - **Read Time**: 30 minutes
   - **Contains** (10 sections):
     1. Domain and problem setup (continental rifting diagram)
     2. Data structure overview (class hierarchy)
     3. Staggered grid layout (ASCII diagrams)
     4. Complete time loop sequence (flowchart)
     5. Non-linear iteration details (Picard/Newton flowchart)
     6. Rheology models (combined viscosity)
     7. Marker operations and lifecycle
     8. Grid ↔ Marker information flow
     9. Performance bottlenecks & solutions
     10. Final code organization structure
   - **Action**: Reference when designing classes and APIs

---

### 5. **SPECKIT_PROMPT_DESIGN.md** ← IMPLEMENTATION ROADMAP
   - **Purpose**: Guide for using Speckit to build modules
   - **Read Time**: 40 minutes
   - **Contains** (10 sections):
     1. Speckit fundamentals for this project
     2. Prompt template structure
     3. Concrete prompt examples:
        - Phase 1: StokesGrid, Material, Marker classes (ready to use!)
        - Phase 2: Interpolation & MarkerSwarm
        - Phase 3: Matrix assembly, non-linear solver
        - Phase 4: Time stepper & main loop
     4. Best practices for prompting
     5. Implementation checklist
     6. Recommended execution order
     7. Key prompting tips
     8. Questions to refine before submission
   - **Action**: Use as template for your Speckit prompts

---

## 🎯 Quick Navigation by Task

### "I want to understand what SiSteR does..."
→ Read **README_KNOWLEDGE_BUILD.md** (executive summary)
→ Then **SISTER_KNOWLEDGE_CONTEXT.md** (parts 1-3)

### "I need to implement StokesGrid..."
→ **SPECKIT_PROMPT_DESIGN.md** (Part 3 - Prompt 1A)
→ **ARCHITECTURE_VISUAL_GUIDE.md** (section 3 - grid layout)
→ **SISTER_KNOWLEDGE_CONTEXT.md** (part 3.1 - staggered grid)

### "I'm writing the FD matrix assembly..."
→ **STOKES_MATHEMATICS.md** (sections 3-5)
→ **ARCHITECTURE_VISUAL_GUIDE.md** (section 2 - data flow)
→ **SPECKIT_PROMPT_DESIGN.md** (Part 5 - Prompt 3A)

### "I want to understand Picard/Newton iterations..."
→ **SISTER_KNOWLEDGE_CONTEXT.md** (part 3.3)
→ **ARCHITECTURE_VISUAL_GUIDE.md** (section 5)
→ **STOKES_MATHEMATICS.md** (section 7)
→ **SPECKIT_PROMPT_DESIGN.md** (Part 5 - Prompt 3B)

### "How do markers work?"
→ **SISTER_KNOWLEDGE_CONTEXT.md** (part 5)
→ **ARCHITECTURE_VISUAL_GUIDE.md** (section 7-8)
→ **SPECKIT_PROMPT_DESIGN.md** (Part 4 - Prompt 2A)

### "I'm setting up the time loop..."
→ **ARCHITECTURE_VISUAL_GUIDE.md** (section 4)
→ **SISTER_KNOWLEDGE_CONTEXT.md** (part 3.4)
→ **SPECKIT_PROMPT_DESIGN.md** (Part 6 - Prompt 4A)

### "What's the continental rifting example?"
→ **SISTER_KNOWLEDGE_CONTEXT.md** (part 6)
→ **ARCHITECTURE_VISUAL_GUIDE.md** (section 1)
→ **SiStER-master/SiStER_Input_File_continental_rift.m**

### "How do I test my code?"
→ **README_KNOWLEDGE_BUILD.md** (success criteria)
→ **SPECKIT_PROMPT_DESIGN.md** (acceptance criteria in each prompt)
→ **STOKES_MATHEMATICS.md** (section 9 - analytical solutions)

---

## 📋 Document Statistics

| Document | Pages | Sections | Use |
|----------|-------|----------|-----|
| README_KNOWLEDGE_BUILD.md | 6 | 10 | **Quick start** |
| SISTER_KNOWLEDGE_CONTEXT.md | 25 | 9 | **Main reference** |
| STOKES_MATHEMATICS.md | 20 | 10 | **Technical details** |
| ARCHITECTURE_VISUAL_GUIDE.md | 15 | 10 | **Design guide** |
| SPECKIT_PROMPT_DESIGN.md | 18 | 10 | **Implementation guide** |
| **TOTAL** | **~84** | | |

---

## 🔄 Recommended Reading Path

### For Complete Understanding (3-4 hours)
1. **README_KNOWLEDGE_BUILD.md** (20 min) — Big picture
2. **SISTER_KNOWLEDGE_CONTEXT.md** (45 min) — What is SiSteR?
3. **ARCHITECTURE_VISUAL_GUIDE.md** (30 min) — Visual design
4. **SPECKIT_PROMPT_DESIGN.md** (40 min) — Implementation strategy
5. **STOKES_MATHEMATICS.md** (60 min, optional) — Deep math

### For Implementation (Ongoing Reference)
- Bookmark **SISTER_KNOWLEDGE_CONTEXT.md** for physics/algorithm questions
- Bookmark **STOKES_MATHEMATICS.md** for equation/stencil questions
- Bookmark **SPECKIT_PROMPT_DESIGN.md** when writing prompts
- Bookmark **ARCHITECTURE_VISUAL_GUIDE.md** for API design questions

### For Debugging/Optimization (As Needed)
- **ARCHITECTURE_VISUAL_GUIDE.md** section 9 (bottlenecks)
- **STOKES_MATHEMATICS.md** section 8 (stability/scaling)
- **SISTER_KNOWLEDGE_CONTEXT.md** part 7 (challenges)

---

## 🚀 Getting Started: Next 3 Steps

### Step 1: Orient Yourself (Today, 30 min)
- [ ] Skim README_KNOWLEDGE_BUILD.md
- [ ] Read SISTER_KNOWLEDGE_CONTEXT.md parts 1-3
- [ ] Ask clarifying questions

### Step 2: Design Your Architecture (This Week, 2-3 hours)
- [ ] Read ARCHITECTURE_VISUAL_GUIDE.md completely
- [ ] Draw class diagrams based on section 10
- [ ] Decide: NumPy-only? Numba? GPU?
- [ ] Set up project structure (git, pyproject.toml, tests/)

### Step 3: Write Your First Speckit Prompt (Next Week)
- [ ] Pick ONE module: StokesGrid (easiest start)
- [ ] Read relevant prompt from SPECKIT_PROMPT_DESIGN.md
- [ ] Refine it with specifics from MATLAB code:
  - Index ordering from `SiStER_Initialize.m`
  - Parameter ranges from input files
  - Example usage from main loop
- [ ] Submit to Speckit with:
  - Context (2 paragraphs)
  - Specification (1 paragraph)
  - Requirements (5-10 bullets)
  - Constraints (3-5 bullets)
  - Acceptance criteria (3-5 tests)
  - Example usage (code snippet)

---

## 📝 Quick Reference: Key Concepts

### Grid
- **Staggered MAC grid**: pressure at corners, velocity at edges
- **Why**: Avoids pressure oscillations, natural divergence-free
- **Coordinates**: Normal nodes (i,j), shear nodes (i±1/2, j±1/2)

### Stokes Equations
- **∇p = η∇²v + ρg** (momentum)
- **∇·v = 0** (continuity)
- **Solved together** as coupled system (not sequential)

### Viscosity Models
- **Ductile**: η ∝ ε̇^(1-n)/n · exp(E/nRT) (temperature/stress dependent)
- **Plastic**: η capped at σ_Y/(2ε̇) (Mohr-Coulomb)
- **Elastic**: σ = σ_old + 2G·Δε (memory)
- **Combined**: η_eff = min(η_ductile, η_plastic)

### Iteration Strategy
- **Picard (robust)**: S_new = L⁻¹R (uses current viscosity)
- **Newton (fast)**: S_new = S - L⁻¹(LS - R) (quadratic convergence)
- **Convergence**: ||LS - R||₂ / ||R||₂ < 1e-9 (L2 residual norm)

### Markers & Advection
- **Purpose**: Track material history (stress, strain, composition)
- **Operations**: Interpolate to/from grid, advect with velocity
- **Reseed**: Maintain ~10 markers per grid cell
- **Feedback**: Provide material properties to grid for next solve

### Time Loop
1. Interpolate markers → grid properties
2. Solve Stokes (Picard/Newton) → get v, p
3. Interpolate strain rate → markers
4. Update marker stresses
5. Advect markers
6. Reseed
7. Repeat

---

## ✅ Knowledge Base Checklist

Before you start coding, verify you understand:

### Physics
- [ ] Why Stokes equations (not Navier-Stokes)
- [ ] What is strain rate tensor
- [ ] How viscosity depends on temperature/stress
- [ ] Difference between ductile, plastic, elastic
- [ ] Why markers are needed (large deformation)

### Numerics
- [ ] What is staggered grid (why used)
- [ ] Finite difference stencils (4-point, 9-point)
- [ ] Matrix assembly (loop structure, indexing)
- [ ] Non-linear iteration (Picard vs Newton)
- [ ] Scaling for numerical stability (Kc, Kb)

### Algorithm
- [ ] Time loop sequence (9-11 steps)
- [ ] Picard iteration (when to use)
- [ ] Newton iteration (when to switch)
- [ ] Convergence criteria (residual norm)
- [ ] Marker advection (CFL condition)

### Code Design
- [ ] Class hierarchy (Grid, Material, Solver, Simulation)
- [ ] Interfaces between classes (what each provides)
- [ ] Data structures (arrays, shapes, dtypes)
- [ ] I/O format (HDF5 for markers, netCDF for fields)
- [ ] Testing strategy (unit → integration → regression)

---

## 📞 Getting Help

### If you're confused about...

**"What is SiSteR for?"**
→ SISTER_KNOWLEDGE_CONTEXT.md, Part 1

**"How does the Stokes solver work?"**
→ SISTER_KNOWLEDGE_CONTEXT.md, Part 3 + STOKES_MATHEMATICS.md

**"How should I structure my code?"**
→ ARCHITECTURE_VISUAL_GUIDE.md + SPECKIT_PROMPT_DESIGN.md

**"What should my first Speckit prompt be?"**
→ SPECKIT_PROMPT_DESIGN.md, Part 3

**"What are the equations I need to implement?"**
→ STOKES_MATHEMATICS.md, Sections 1-5

**"How do I test if my code is correct?"**
→ STOKES_MATHEMATICS.md, Section 9 + README_KNOWLEDGE_BUILD.md

**"What's the continental rifting example?"**
→ SISTER_KNOWLEDGE_CONTEXT.md, Part 6

**"What are common pitfalls?"**
→ SISTER_KNOWLEDGE_CONTEXT.md, Part 7 + STOKES_MATHEMATICS.md, Section 8

---

## 🎓 Learning Resources (External)

### Theory
- **Gerya & Yuen (2003)**: Marker-in-cell method (foundational)
- **Turcotte & Schubert (2014)**: Geodynamics textbook
- **Elman et al. (2014)**: Stokes solvers and saddle-point systems

### Numerics
- **SciPy Documentation**: `scipy.sparse.linalg` for solvers
- **NumPy Documentation**: Array operations and broadcasting
- **Numba Documentation**: JIT compilation for Python

### Similar Projects
- **Underworld** (Python, geodynamics)
- **ASPECT** (C++, finite element method)
- **code_aster** (Finite element, open source)

---

## 📦 What You Have

✅ **Complete physics understanding** (from 5 docs)
✅ **Clear algorithm documentation** (flowcharts, pseudocode)
✅ **Concrete code examples** (Speckit prompts ready to use)
✅ **Design patterns** (OOP structure for Python)
✅ **Validation strategy** (test cases, analytical solutions)
✅ **Performance roadmap** (bottlenecks identified, solutions proposed)

## 🎯 What You Need to Do

1. **Read** the documents (3-4 hours)
2. **Design** your architecture (2-3 hours)
3. **Set up** project structure (1 hour)
4. **Write** first Speckit prompt (1-2 hours)
5. **Iterate** through 5 phases (2-3 months)

---

## 🏁 Success Definition

You'll know the knowledge base is complete and useful when:

- ✅ You can explain "what is SiSteR" in 2 minutes
- ✅ You understand why staggered grid is used
- ✅ You can draw the time loop from memory
- ✅ You know how to validate a Stokes solver (analytical test)
- ✅ You can design a Speckit prompt for any module
- ✅ You can answer "how does X component talk to Y component"

**If you can't do all of the above**, spend more time with the documents.

**If you can do all of the above**, you're ready to start implementing! 🚀

---

## 📄 File Locations

All documents are in: `c:\Users\AntoineDemont\Desktop\Perso\Git_projects\SiSteR-py\`

- `README_KNOWLEDGE_BUILD.md` ← Start here
- `SISTER_KNOWLEDGE_CONTEXT.md` ← Main reference
- `STOKES_MATHEMATICS.md` ← Technical details
- `ARCHITECTURE_VISUAL_GUIDE.md` ← Design guide
- `SPECKIT_PROMPT_DESIGN.md` ← Implementation guide
- `SiStER-master/` ← Original MATLAB code

---

## 💡 Pro Tips

1. **Print or PDF** the documents for reference while coding
2. **Bookmark key sections** in your browser/editor
3. **Cross-reference** documents when confused (they're designed to complement each other)
4. **Review MATLAB code** alongside docs to see actual implementation
5. **Keep notes** on decisions (why you chose architecture X over Y)
6. **Run continental rifting example** once you have basic structure working

---

## 🎉 Final Note

You now have a **comprehensive knowledge base** built on:
- Direct analysis of the SiSteR MATLAB codebase
- Deep understanding of Stokes equations and numerics
- Clear Python/OOP design patterns
- Ready-to-use Speckit prompts
- Validation and testing strategies
- Performance optimization roadmap

**The hard part (research and design) is done.**

**The fun part (implementation) is about to begin!**

Good luck with SiSteR-py! 🚀

---

*Knowledge base created: December 2024*
*Total time invested: Research & documentation*
*Ready for implementation: Yes ✓*
