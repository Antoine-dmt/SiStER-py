# SiSteR Architecture: Visual & Conceptual Guide

## 1. Domain & Problem Setup

```
CONTINENTAL RIFTING EXAMPLE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    x (0 to 170 km)
    ├─────────────────────────────────────────┤
    
0 m ┌─────────────────────────────────────────┐ ↑
    │ STICKY LAYER (Phase 1)                  │ |
    │ ρ = 1000 kg/m³, low viscosity           │ | 10 km
10k │ ┌─────────────────────────────────────┐ │ ↓
    │ │ WEAK FAULT ZONE (60° dip)           │ │
    │ │ 1 km width, reduced friction        │ │
    │ │                                     │ │
    │ │ LITHOSPHERE/MANTLE (Phase 2)        │ |
60k │ │ ρ = 3300 kg/m³, strong creep        │ | 50 km
    │ │ Power-law dislocation, n=3.5        │ |
    │ │                                     │ │
    │ └─────────────────────────────────────┘ │
    └─────────────────────────────────────────┘

BOUNDARY CONDITIONS:
  Top: Velocity extension (1 cm/yr = 3e-10 m/s)
  Bottom: Fixed (no flow)
  Sides: Prescribed extension
  
FORCES: Gravity (gy = 9.8 m/s²) + boundary motion
```

---

## 2. Data Structure Overview

```
                    ┌──────────────────────┐
                    │   GeodynamicsSimulation
                    │   (Main Controller)  │
                    └──────┬───────────────┘
                           │
          ┌────────────────┼────────────────┐
          ▼                ▼                ▼
    ┌──────────┐     ┌──────────┐     ┌──────────┐
    │ Grid     │     │Materials │     │Markers   │
    ├──────────┤     ├──────────┤     ├──────────┤
    │• x, y    │     │• phases  │     │• xm, ym  │
    │• Nx, Ny  │     │• density │     │• im      │
    │• interp  │     │• rheology│     │• sxxm    │
    └──┬───────┘     └──┬───────┘     └──┬───────┘
       │                │               │
       │      ┌─────────▼──────────┐   │
       │      │ StokesFlow         │   │
       │      ├────────────────────┤   │
       │      │• Assemble L, R     │   │
       │      │• Solve (Picard/NW) │   │
       │      │• Extract v, p      │   │
       │      └────────────────────┘   │
       │                               │
       └──────────┬────────────────────┘
                  │
        Time Stepping Loop:
        
        for t = 1 to Nt:
            1. Markers → Nodes (interpolate properties)
            2. Solve Stokes (get v, p)
            3. Nodes → Markers (interpolate strain rate)
            4. Update marker stresses
            5. OUTPUT (if requested)
            6. Advect markers
            7. Next time step
```

---

## 3. Staggered Grid Layout (2D)

```
GRID NODES (Nx=3, Ny=3)
═══════════════════════

         j=0    j=1    j=2    j=3    (X-index)
       
i=0  ┌──•────┬──•────┬──•────┬──•──┐
     │  P(0,0) P(0,1) P(0,2) P(0,3) │
  y=0├───────┼───────┼───────┼──────┤
     │ v_x ◊  v_y  v_x ◊  v_y      │
     │    (0,0.5)(0.5,0)(1,0.5)    │
i=1  ├──•────┬──•────┬──•────┬──•──┤
     │ σ_xx σ_xy    σ_xy            │
     │    (0.5,0.5)                │
     │  P(1,0) P(1,1) P(1,2) P(1,3) │
     │ v_x ◊  v_y  v_x ◊          │
i=2  ├──•────┬──•────┬──•────┬──•──┤
     │                                
i=3  └──•────┴──•────┴──•────┴──•──┘
   
LEGEND:
  •    = Normal node (pressure)
  ◊    = Shear node (velocity, stress)
  y ↑  (depth, actually)
  x →

SOLUTION VECTOR (example, small grid):
S = [p(0,0), vx(0,0), vy(0,0),  p(0,1), vx(0,1), vy(0,1), ...]

For each point (i,j), store 3 components: [p, vx, vy]
Linear index k = 3*((j-1)*Ny + i) - 2  (for pressure)
```

---

## 4. Time Loop Sequence

```
TIME STEP t:
════════════

                    ▼
        ┌───────────────────────┐
        │ Start of iteration t  │
        └───────────┬───────────┘
                    ▼
        ┌───────────────────────────────────────┐
        │ 1. Interpolate marker data to nodes   │
        │                                       │
        │    • Phase from markers → nodes       │
        │    • Density from markers → nodes     │
        │    • Stress history from markers      │
        │    • Temperature (if thermal solve)   │
        │                                       │
        │  Output: ρ(grid), T(grid), etc.      │
        └───────────┬───────────────────────────┘
                    ▼
        ┌───────────────────────────────────────┐
        │ 2. SOLVE STOKES EQUATIONS             │
        │                                       │
        │    FOR iteration = 1 to Npicard:     │
        │      • Compute viscosity η(ε̇, T, σ)  │
        │      • Assemble FD matrix L           │
        │      • Assemble RHS vector R          │
        │      • Solve: L·S = R                 │
        │      • Check convergence:             │
        │        ||L·S - R||₂ / ||R||₂ < tol   │
        │      • If converged, break            │
        │                                       │
        │  Output: vx, vy, p (on grid)         │
        │  Output: ε̇xx, ε̇yy, ε̇xy (strain rate)│
        └───────────┬───────────────────────────┘
                    ▼
        ┌───────────────────────────────────────┐
        │ 3. Interpolate strain rate to markers │
        │                                       │
        │    ε̇(markers) = interp(ε̇(nodes))     │
        │                                       │
        │  Output: ε̇_II on markers             │
        └───────────┬───────────────────────────┘
                    ▼
        ┌───────────────────────────────────────┐
        │ 4. Update marker stresses             │
        │                                       │
        │    σⁿ⁺¹ = σⁿ + Δσ(ε̇, Δt)             │
        │                                       │
        │    • Elastic: σ = σ_old + 2G·Δε_elast │
        │    • Rotation: σ → R(ω)·σ·R(ω)ᵀ      │
        │    • Plasticity: σ → min(σ, σ_yield) │
        │                                       │
        │  Output: sxxm, sxym on markers       │
        └───────────┬───────────────────────────┘
                    ▼
        ┌───────────────────────────────────────┐
        │ 5. Update plastic strain              │
        │    (only if plasticity enabled)       │
        │                                       │
        │    e_p ← e_p + Δe_p(σ, ε̇)           │
        │                                       │
        │  Output: ep (cumulative strain)       │
        └───────────┬───────────────────────────┘
                    ▼
        ┌───────────────────────────────────────┐
        │ 6. OUTPUT (if mod(t, dt_out) == 0)   │
        │                                       │
        │    Save to file:                      │
        │    • vx(t), vy(t) — velocity         │
        │    • p(t) — pressure                 │
        │    • T(t) — temperature              │
        │    • η(t) — viscosity                │
        │    • ε̇_II(t) — strain rate          │
        │    • σxx, σxy — stress               │
        │    • Phase map                       │
        │    • Markers: xm, ym, sxx, ep       │
        └───────────┬───────────────────────────┘
                    ▼
        ┌───────────────────────────────────────┐
        │ 7. Set adaptive time step             │
        │                                       │
        │    Δt = 0.5 · min(Δx, Δy) / v_max   │ (CFL)
        │                                       │
        │  Output: dt_m (time step for next)    │
        └───────────┬───────────────────────────┘
                    ▼
        ┌───────────────────────────────────────┐
        │ 8. Rotate elastic stresses            │
        │    (only if elasticity enabled)       │
        │                                       │
        │    σ(t+Δt) = R(ω·Δt)·σ(t)·R(ω·Δt)ᵀ  │
        │                                       │
        │    where ω = rotation rate            │
        │                                       │
        │  Output: rotated σxx, σxy            │
        └───────────┬───────────────────────────┘
                    ▼
        ┌───────────────────────────────────────┐
        │ 9. THERMAL DIFFUSION                  │
        │    (only if Tsolve enabled)           │
        │                                       │
        │    ∂T/∂t = κ·∇²T  (heat equation)    │
        │                                       │
        │  Output: T(t+Δt)                     │
        └───────────┬───────────────────────────┘
                    ▼
        ┌───────────────────────────────────────┐
        │ 10. MARKER ADVECTION                  │
        │     (Lagrangian step)                 │
        │                                       │
        │    x_m(t+Δt) = x_m(t) + v(x_m) · Δt  │
        │                                       │
        │    For each marker:                   │
        │    • Interpolate v from grid          │
        │    • Move marker: x_m ← x_m + v·Δt   │
        │    • Check bounds (remove if outside) │
        │                                       │
        │  Output: xm, ym (updated positions)   │
        └───────────┬───────────────────────────┘
                    ▼
        ┌───────────────────────────────────────┐
        │ 11. RESEED MARKERS                    │
        │                                       │
        │    Where marker density < threshold:  │
        │    • Add new markers uniformly        │
        │    • Inherit phase & stress from     │
        │      interpolation of neighbors       │
        │                                       │
        │  Output: new markers added            │
        └───────────┬───────────────────────────┘
                    ▼
        ┌───────────────────────────────────────┐
        │ End of iteration t                    │
        │ time ← time + Δt                      │
        └───────────┬───────────────────────────┘
                    ▼
        ┌───────────────────────────────────────┐
        │ if t < Nt, goto next iteration        │
        └───────────────────────────────────────┘
```

---

## 5. Non-Linear Iteration (Inside Step 2)

```
PICARD/NEWTON ITERATION
═══════════════════════

Initialize: S₀ = initial guess (or from last time step)

FOR pit = 1 to Npicard_max:
    
    ┌─────────────────────────────────────┐
    │ Compute viscosity at all nodes      │
    │                                     │
    │ From current S (v, p):              │
    │   ε̇ = compute_strain_rate(v)       │
    │   σ = compute_stress(p, ε̇, state)  │
    │   η = viscosity(ε̇, σ, T)           │
    │       ├─ Dislocation: η ∝ ε̇ˠ⁻¹    │
    │       ├─ Diffusion: η ∝ ε̇⁰         │
    │       └─ Plastic: min(η, σ_Y/ε̇)   │
    └──────────┬──────────────────────────┘
               ▼
    ┌─────────────────────────────────────┐
    │ Assemble system (depends on η)      │
    │                                     │
    │ L(η) = Finite difference matrix     │
    │ R = Right-hand side (gravity + BC)  │
    │                                     │
    │ Both depend on current η!           │
    └──────────┬──────────────────────────┘
               ▼
    ┌─────────────────────────────────────┐
    │ Compute residual                    │
    │                                     │
    │ Res = L·Sₚᵢₜ - R                    │
    │ L2norm = ||Res||₂ / ||R||₂          │
    │                                     │
    └──────────┬──────────────────────────┘
               ▼
              ┌─ Is pit == 1?
              │
        YES ──┤→ Picard update:
              │   Sₚᵢₜ₊₁ = L⁻¹·R
              │
        NO  ──┤→ Is pit >= Npicard_switch?
                 │
           YES ─→ Newton update:
                 │ Sₚᵢₜ₊₁ = Sₚᵢₜ - (L⁻¹·Res)
                 │
           NO  ─→ Picard update:
                   Sₚᵢₜ₊₁ = L⁻¹·R
                   
               ▼
    ┌─────────────────────────────────────┐
    │ Check convergence                   │
    │                                     │
    │ if L2norm < tolerance AND           │
    │    pit >= Npicard_min:              │
    │    CONVERGED ✓                      │
    │    Break loop                       │
    │                                     │
    │ else if pit == Npicard_max:         │
    │    WARNING: Max iterations reached  │
    │    Break loop anyway                │
    │                                     │
    │ else:                               │
    │    Continue to pit+1                │
    └─────────────────────────────────────┘

Result: S_final = [p, vx, vy] converged solution
```

---

## 6. Rheology Models

```
VISCOSITY COMPUTATION
═════════════════════

Three parallel models combined (harmonic mean):

1. DUCTILE CREEP (temperature-dependent power law)
   ────────────────────────────────────────────────
   
   Power law: ε̇ᵢᵢ = A·σⁿ·exp(-E/nRT)
   
   Effective viscosity: η_ductile = σᵢᵢ / (2·ε̇ᵢᵢ)
   
                      = σᵢᵢ / (2·A·σⁿ·exp(-E/nRT))
   
   σ ↑ → ε̇ ↑ → η ↓ (higher stress → weaker)
   T ↑ → η ↓ (hotter → weaker)
   
   Dislocation creep: n ~ 3 (strong stress dependence)
   Diffusion creep: n = 1 (linear stress)
   

2. PLASTICITY (Mohr-Coulomb yield criterion)
   ───────────────────────────────────────────
   
   Yield strength: σ_Y = (C + μ·P)·cos(arctan(μ))
   
   If σᵢᵢ > σ_Y:
       η_plastic = σ_Y / (2·ε̇ᵢᵢ)  ← Capped!
   
   C = cohesion (pressure-independent strength)
   μ = friction coefficient
   P = pressure
   
   Higher pressure → higher yield strength (confining effect)
   

3. ELASTICITY (Stress accumulation)
   ────────────────────────────────
   
   Elastic stress: σ_elastic = 2·G·ε_elastic
   
   Total strain: ε_total = ε_elastic + ε_viscous + ε_plastic
   
   Stress evolves: σⁿ⁺¹ = σⁿ + 2·G·Δε_viscous - σⁿ·∇·v·Δt
   
   Elastic memory: stresses "remember" past deformation


EFFECTIVE VISCOSITY (Combined)
──────────────────────────────

η_eff = min(η_ductile, η_plastic)

         ∨
    ┌────────────────────┐
    │  Powers within     │
    │  range [ηmin,    │ 
    │         ηmax]    │
    │                  │
    └────────────────────┘
         ∨
    Used in assembly: σ = 2·η_eff·ε̇ - p·I
```

---

## 7. Marker Operations

```
MARKER SWARM LIFECYCLE
══════════════════════

INITIALIZATION:
───────────────
                    ┌─────────────────────┐
                    │ Create uniform grid │
                    │ Mquad markers/cell  │
                    └──────────┬──────────┘
                              ▼
                    ┌─────────────────────┐
                    │ Assign phases by    │
                    │ geometry (layer id) │
                    └──────────┬──────────┘
                              ▼
                    ┌─────────────────────┐
                    │ Initialize stresses │
                    │ to zero             │
                    │ Temperature from    │
                    │ geotherm            │
                    └─────────────────────┘
                    

EACH TIME STEP (Marker perspective):
───────────────────────────────────

    ┌──────────────────────────┐
    │ Step 1: Get velocity at  │
    │ marker location          │
    │                          │
    │ v_m = interp_grid(v_m,  │
    │       x_m, y_m)          │
    └──────────┬───────────────┘
              ▼
    ┌──────────────────────────┐
    │ Step 2: Advect marker    │
    │                          │
    │ x_new = x_old + v·Δt    │
    │                          │
    │ Note: Material property  │
    │ moves with marker!       │
    └──────────┬───────────────┘
              ▼
    ┌──────────────────────────┐
    │ Step 3: Accumulate       │
    │ stress evolution         │
    │                          │
    │ σ_new = σ_old + Δσ      │
    │                          │
    │ Update stress history    │
    └──────────┬───────────────┘
              ▼
    ┌──────────────────────────┐
    │ Step 4: Check position   │
    │                          │
    │ if outside domain:       │
    │   Remove marker          │
    │                          │
    │ if in low-density region │
    │   Mark for reseeding     │
    └──────────┬───────────────┘
              ▼
    ┌──────────────────────────┐
    │ Step 5: Interpolate to   │
    │ grid for next iteration  │
    │                          │
    │ ρ(grid) = avg(ρ_marker) │
    │ T(grid) = avg(T_marker) │
    │                          │
    │ (Weighted by marker pos) │
    └──────────────────────────┘


RESEED STRATEGY:
────────────────

    Loop over grid cells:
        
        Density = # markers / cell volume
        
        if Density < Mquad_crit:
            ✗ Too few markers!
            
            → Add new markers uniformly
            → Copy phase from neighbors
            → Interpolate stress from neighbors
        
        if Density > 2·Mquad:
            ✗ Too many markers!
            
            → Remove excess markers randomly
            → (Optional: reduce computational cost)
```

---

## 8. Coupling: Grid ↔ Markers

```
INFORMATION FLOW
════════════════

Time step t:
           
    MARKERS (Lagrangian)          GRID (Eulerian)
    ═════════════════════         ════════════════
    
    xm, ym (positions)
    im (phase)                    
    sxxm, sxym (stresses)
    em (plastic strain)
    Tm (temperature)
              │
              │  INTERPOLATE TO GRID
              ▼
              ┌──────────────────────┐
              │ Phase → nodes        │
              │ Density → nodes      │
              │ Temperature → nodes  │
              │ Stress history      │
              └──────────┬───────────┘
                        ▼
                   ρ(grid)
                   T(grid)
                   phase(grid)
                        ▼
                   ┌──────────────┐
                   │  SOLVE       │
                   │  STOKES      │
                   │              │
                   │ → v, p, ε̇    │
                   └──────┬───────┘
                         ▼
              ┌──────────────────────┐
              │ Strain rate → markers│
              │ Velocity → markers   │
              │ (for advection)      │
              └──────────┬───────────┘
                        ▼
    ε̇m, vm (on markers now)
              │
              │  UPDATE MARKER STATE
              ▼
    σm = σm + Δσ(ε̇m)
    xm = xm + vm·Δt
    
    (Back to top of loop)
```

---

## 9. Performance Bottlenecks & Solutions

```
WHAT'S SLOW?              WHY?                      FIX
════════════════════════════════════════════════════════════════════

1. Matrix assembly        Loop through all 3×Nx×Ny points,
                          compute stencils                  → Numba JIT
                                                            → Vectorize
                          
2. Picard iterations      ~10-100 solves per time step
                          Each solve: 100-1000 seconds     → Preconditioner
                                                            → GPU sparse solve
                          
3. Interpolation          Bilinear interp to/from markers
                          ~100k-1M markers × many interp   → Numba JIT
                                                            → KDTree for queries
                          
4. Marker advection       Move all markers forward
                          + interpolate velocities          → Vectorize
                                                            → NumPy broadcast
                          
5. Viscosity update       Compute η from ε̇ for all nodes
                          per Picard iteration              → NumPy vectorize
                                                            → Lookup tables

TARGET: 1000 time steps in < 1 hour (feasible with optimization)
```

---

## 10. Code Organization (Final)

```
sister-py/
├── src/
│   └── sister/
│       ├── __init__.py
│       │
│       ├── grid.py          ← StokesGrid class
│       ├── material.py       ← Material, Rheology classes
│       ├── marker.py         ← Marker, MarkerSwarm classes
│       │
│       ├── assembly.py       ← StokesMatrixAssembler
│       ├── solver.py         ← StokesNonlinearSolver
│       ├── flow.py           ← StokesFlow (orchestrator)
│       │
│       ├── simulation.py     ← GeodynamicsSimulation (main loop)
│       ├── io.py             ← Save/load, HDF5 I/O
│       ├── visualization.py  ← Plotting utilities
│       │
│       └── utils/
│           ├── interpolation.py
│           ├── stress.py
│           ├── rheology.py
│           └── constants.py
│
├── tests/
│   ├── test_grid.py
│   ├── test_material.py
│   ├── test_marker.py
│   ├── test_assembly.py
│   ├── test_solver.py
│   ├── test_integration.py
│   └── data/
│       └── matlab_reference/  ← Expected outputs from MATLAB
│
├── examples/
│   ├── continental_rifting.py      ← Default example
│   ├── shear_flow.py               ← Simple validation case
│   └── gravity_column.py            ← Hydrostatic test
│
├── docs/
│   ├── architecture.md
│   ├── tutorial.md
│   └── physics.md
│
├── pyproject.toml
├── README.md
└── LICENSE
```

---

## Summary

```
SiSteR-py Architecture: 4 Core Layers
═════════════════════════════════════

Layer 1: DATA STRUCTURES (Grid, Material, Marker)
    └─ What is the problem? Where are things?

Layer 2: SOLVERS (MatrixAssembler, NonlinearSolver)
    └─ How do we solve the physics?

Layer 3: INTEGRATION (StokesFlow, GeodynamicsSimulation)
    └─ How do we evolve through time?

Layer 4: APPLICATIONS (Examples, Visualizations)
    └─ How do we use this to science?


Key Insight:
  Each layer depends on below, independent from above
  → Can test each layer separately
  → Can swap implementations (e.g., CPU → GPU)
  → Clear interfaces between components
```

This design will make SiSteR-py maintainable, fast, and extensible! 🚀
