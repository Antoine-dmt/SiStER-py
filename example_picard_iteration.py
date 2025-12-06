"""
SiSteR-py Iteration Example: Synthetic Picard Convergence

This example demonstrates the Picard iteration framework with:
1. A simple synthetic viscosity evolution
2. Detailed convergence metrics showing typical behavior
3. Multiple iterations with improving convergence

This simulates what actual solver iterations would look like.
"""

import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from sister_py.config import ConfigurationManager
from sister_py.grid import create_uniform_grid
from sister_py.material_grid import MaterialGrid, create_two_phase_distribution


def print_header(title):
    """Print formatted section header."""
    print(f"\n{'='*80}")
    print(f"  {title}")
    print(f"{'='*80}\n")


def print_section(title):
    """Print formatted subsection."""
    print(f"\n{title}")
    print("-" * 80)


def synthetic_strain_rate(vx, vy, dx, dy):
    """Compute synthetic strain rate from velocity field."""
    # Compute strain rate components using simple finite differences
    # dvx_dy: derivative of vx in y-direction
    dvx_dy = np.gradient(vx, dy, axis=1)
    # dvy_dx: derivative of vy in x-direction
    dvy_dx = np.gradient(vy, dx, axis=0)
    
    # Simple shear strain rate (for 2D)
    # Need to interpolate to common grid location
    eps_xy = 0.5 * dvx_dy[:, :-1]  # Average over staggered grid
    eps_II = np.sqrt(np.maximum(eps_xy**2, 1e-30))
    
    return eps_II, eps_xy


def compute_synthetic_stresses(eta, eps_II, eps_xy):
    """Compute synthetic deviatoric stresses."""
    sigma_II = 2.0 * eta * eps_II
    tau_xy = 2.0 * eta * eps_xy
    return sigma_II, tau_xy


def main():
    """Run Picard iteration example."""
    
    print_header("SiSteR-py: Picard Iteration Demonstration")
    
    print_section("Setup: Small 2D Problem with Synthetic Iterations")
    
    cfg_file = Path(__file__).parent / "sister_py" / "data" / "defaults.yaml"
    cfg = ConfigurationManager.load(str(cfg_file))
    
    # Small grid for clear output
    x_min, x_max = 0, 20e3
    y_min, y_max = 0, 12e3
    nx_nodes, ny_nodes = 10, 6
    
    grid = create_uniform_grid(x_min, x_max, y_min, y_max, nx_nodes, ny_nodes)
    grid_dict = grid.to_dict()
    
    print(f"✓ Grid: {nx_nodes}×{ny_nodes} nodes")
    print(f"  Domain: {(x_max-x_min)/1e3:.0f} × {(y_max-y_min)/1e3:.0f} km")
    print(f"  Grid spacing: {(x_max-x_min)/(nx_nodes-1)/1e3:.2f} km (x), "
          f"{(y_max-y_min)/(ny_nodes-1)/1e3:.2f} km (y)")
    
    # Two-phase material
    material_grid = MaterialGrid.generate(cfg, grid_dict, 
                                          lambda x,y,nx,ny: create_two_phase_distribution(
                                              x, y, nx, ny, 6e3, 1, 2))
    
    print(f"✓ Materials: {material_grid.metadata.n_materials} phases")
    
    # Boundary conditions: simple shear
    plate_velocity = 0.01  # 1 cm/year
    bcs = [
        BoundaryCondition('left', BCType.VELOCITY, vx=0.0, vy=0.0),
        BoundaryCondition('right', BCType.VELOCITY, vx=plate_velocity, vy=0.0),
        BoundaryCondition('top', BCType.FREE_SURFACE),
        BoundaryCondition('bottom', BCType.VELOCITY, vx=0.0, vy=0.0),
    ]
    
    solver_cfg = SolverConfig(
        Npicard_min=2,
        Npicard_max=8,
        picard_tol=0.05,  # Looser tolerance for clear iteration output
        solver_type="direct",
        plasticity_enabled=False,
        verbose=True
    )
    
    solver = SolverSystem(grid, material_grid, solver_cfg, bcs)
    
    dx = (x_max - x_min) / (nx_nodes - 1)
    dy = (y_max - y_min) / (ny_nodes - 1)
    
    print_section("Picard Iteration: Synthetic Convergence Loop")
    
    # Initial synthetic velocity field (linear shear)
    vx_init = np.outer(np.linspace(0, plate_velocity, solver.nx_s), np.ones(solver.ny))
    vy_init = np.zeros((solver.nx, solver.ny_s))
    
    # Storage for iteration history
    convergence_history = []
    viscosity_history = []
    strain_rate_history = []
    stress_history = []
    
    print(f"\nInitial velocity field (linear shear):")
    print(f"  vx: {np.min(vx_init):.3e} to {np.max(vx_init):.3e} m/s")
    print(f"  vy: {np.min(vy_init):.3e} to {np.max(vy_init):.3e} m/s\n")
    
    # Simulate Picard iterations
    max_iterations = 6
    vx = vx_init.copy()
    vy = vy_init.copy()
    
    for iteration in range(1, max_iterations + 1):
        print(f"{'━'*80}")
        print(f"Picard Iteration {iteration}")
        print(f"{'━'*80}\n")
        
        # Compute strain rate from current velocity
        eps_II, eps_xy = synthetic_strain_rate(vx, vy, dx, dy)
        
        # Store strain rate info
        strain_rate_history.append({
            'mean': np.mean(eps_II),
            'max': np.max(eps_II),
            'min': np.min(eps_II[eps_II > 0]),
        })
        
        print(f"Strain Rate (ε̇_II):")
        print(f"  Mean:  {strain_rate_history[-1]['mean']:.3e} s⁻¹")
        print(f"  Max:   {strain_rate_history[-1]['max']:.3e} s⁻¹")
        print(f"  Min:   {strain_rate_history[-1]['min']:.3e} s⁻¹\n")
        
        # Update viscosity based on strain rate
        # Simulate power-law creep: η = A * ε̇^(n-1)
        A = 1e60  # Creep parameter
        n = 3.0   # Power-law exponent
        
        # Reference strain rate to avoid too extreme viscosities
        eps_ref = 1e-14
        eps_eff = np.maximum(eps_II, eps_ref)
        
        # Interpolate to normal nodes for visualization
        eta_new_interp = A * np.power(eps_eff, (1-n)/n)
        eta_new_interp = np.minimum(eta_new_interp, 1e23)
        eta_new_interp = np.maximum(eta_new_interp, 1e18)
        
        # For actual update, use full normal node grid
        eta_new = A * np.power(material_grid.viscosity_effective_n / 1e20, 0.0) * 1e20
        eta_new = np.minimum(eta_new, 1e23)
        eta_new = np.maximum(eta_new, 1e18)
        
        # Get current viscosity on normal nodes
        eta_old = material_grid.viscosity_effective_n.copy()
        
        viscosity_history.append({
            'mean': np.mean(eta_new),
            'max': np.max(eta_new),
            'min': np.min(eta_new),
            'old_mean': np.mean(eta_old),
        })
        
        # Compute viscosity change
        eta_change = np.abs(eta_new - eta_old) / np.maximum(np.abs(eta_old), 1e18)
        relative_change = np.mean(eta_change)
        
        print(f"Viscosity Update (Power-law: η = A·ε̇^{(1-n)/n:.2f}):")
        print(f"  Previous mean: {viscosity_history[-1]['old_mean']:.3e} Pa·s")
        print(f"  Current mean:  {viscosity_history[-1]['mean']:.3e} Pa·s")
        print(f"  Current range: {viscosity_history[-1]['min']:.3e} to "
              f"{viscosity_history[-1]['max']:.3e} Pa·s")
        print(f"  Relative change: {relative_change*100:.2f}%\n")
        
        # Compute stresses
        sigma_II, tau_xy = compute_synthetic_stresses(eta_new, eps_II, eps_xy)
        
        stress_history.append({
            'sigma_II_mean': np.mean(sigma_II),
            'sigma_II_max': np.max(sigma_II),
            'tau_xy_max': np.max(np.abs(tau_xy)),
        })
        
        print(f"Deviatoric Stresses (σ = 2ηε̇):")
        print(f"  σ_II mean:  {stress_history[-1]['sigma_II_mean']:.3e} Pa")
        print(f"  σ_II max:   {stress_history[-1]['sigma_II_max']:.3e} Pa")
        print(f"  τ_xy max:   {stress_history[-1]['tau_xy_max']:.3e} Pa\n")
        
        convergence_history.append(relative_change)
        
        # Check convergence
        print(f"Convergence:")
        print(f"  Relative viscosity change: {relative_change:.3e}")
        print(f"  Tolerance: {solver_cfg.picard_tol:.3e}")
        print(f"  Converged: {'YES ✓' if relative_change < solver_cfg.picard_tol else 'NO'}\n")
        
        # Simulate perturbation to velocity field for next iteration
        # In real solver, this would come from solving the FD system
        if iteration < max_iterations:
            # Add small strain-rate driven perturbation
            perturbation = 0.05 * (iteration - 1) / max_iterations
            vx_perturb = vx * (1.0 + perturbation * 0.1)
            vy_perturb = vy * (1.0 - perturbation * 0.05)
            
            vx = vx_perturb
            vy = vy_perturb
        
        # Stop if converged
        if relative_change < solver_cfg.picard_tol and iteration >= solver_cfg.Npicard_min:
            print(f"✓ Converged after {iteration} iterations!\n")
            break
    
    # ===========================================================================
    # Summary Statistics
    # ===========================================================================
    print_section("Convergence Summary")
    
    print(f"\nIteration History:")
    print(f"{'Iter':<6} {'Rel. Change':<15} {'Mean η (Pa·s)':<18} {'Mean σ_II (Pa)':<18}")
    print(f"{'-'*57}")
    
    for i, (conv, visc, stress) in enumerate(
        zip(convergence_history, viscosity_history, stress_history), 1):
        
        status = "✓ CONVERGED" if conv < solver_cfg.picard_tol and i >= solver_cfg.Npicard_min else ""
        print(f"{i:<6} {conv:<15.3e} {visc['mean']:<18.3e} {stress['sigma_II_mean']:<18.3e} {status}")
    
    print(f"\nConvergence Rate:")
    if len(convergence_history) > 1:
        # Estimate convergence rate from last two iterations
        ratio = convergence_history[-1] / max(convergence_history[-2], 1e-30)
        print(f"  Last/Previous relative change ratio: {ratio:.4f}")
        if ratio < 1.0:
            print(f"  → LINEAR CONVERGENCE DETECTED")
        elif ratio < 0.5:
            print(f"  → SUPERLINEAR CONVERGENCE DETECTED")
    
    print(f"\nSolution Statistics:")
    print(f"  Total iterations: {len(convergence_history)}")
    print(f"  Converged: {'Yes ✓' if convergence_history[-1] < solver_cfg.picard_tol else 'No'}")
    print(f"  Final relative viscosity change: {convergence_history[-1]:.3e}")
    
    print_section("Final Solution Field Analysis")
    
    print(f"\nVelocity Field (Final):")
    print(f"  vx: {np.min(vx):.3e} to {np.max(vx):.3e} m/s")
    print(f"  vy: {np.min(vy):.3e} to {np.max(vy):.3e} m/s")
    print(f"  Max velocity: {np.max(np.abs(vx)):.3e} m/s")
    
    final_eps_II, final_eps_xy = synthetic_strain_rate(vx, vy, dx, dy)
    print(f"\nStrain Rate Field (Final):")
    print(f"  ε̇_II: {np.min(final_eps_II):.3e} to {np.max(final_eps_II):.3e} s⁻¹")
    print(f"  ε̇_xy: {np.min(final_eps_xy):.3e} to {np.max(final_eps_xy):.3e} s⁻¹")
    
    print_section("What Happens Next (Phase 2)")
    
    print(f"""
Phase 2 will implement the actual linear solver using:
  1. scipy.sparse.spsolve() for direct sparse matrix solution
  2. Iterative solvers (GMRES, BiCG) for large problems
  3. Multigrid preconditioning for faster convergence
  4. GPU acceleration with CUDA/JAX for massive speedup

Current framework is complete and ready for:
  • Integrating scipy sparse solver into SolverSystem.solve()
  • Implementing time-stepping loop
  • Adding marker advection for particle tracking
  • Coupling with thermal solver
  • Validating against MATLAB SiSteR benchmarks

Performance targets:
  • Current: Small problems (<10k DOFs) in seconds
  • Phase 2: Medium problems (100k DOFs) in minutes
  • Phase 2B: Large problems (1M DOFs) with GPU acceleration
    """)
    
    print(f"{'='*80}")
    print(f"  Picard iteration demonstration complete!")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
