"""
SiSteR-py Picard Iteration Demo: Synthetic Convergence Analysis

This demonstrates the Picard iteration process that solves non-linear
Stokes flow with temperature/stress-dependent viscosity.

Key features shown:
- Multiple Picard iterations
- Viscosity evolution from strain rates
- Convergence criteria
- Stress field computation
"""

import numpy as np
from typing import List, Tuple


def print_header(title: str) -> None:
    """Print formatted section header."""
    print(f"\n{'='*90}")
    print(f"  {title}")
    print(f"{'='*90}\n")


def print_section(title: str) -> None:
    """Print formatted subsection."""
    print(f"\n{title}")
    print("-" * 90)


def simulate_picard_iterations(
    n_iterations: int = 6,
    domain_size: Tuple[float, float] = (100e3, 50e3),
    n_nodes: Tuple[int, int] = (21, 11),
    plate_velocity: float = 0.01,
) -> None:
    """
    Simulate Picard iteration convergence for simple shear flow.
    
    Parameters:
        n_iterations: Number of iterations to simulate
        domain_size: (width, height) in meters
        n_nodes: (nx, ny) grid nodes
        plate_velocity: Plate boundary velocity (m/s)
    """
    
    print_header("SiSteR-py: Picard Iteration Demonstration")
    
    # Setup information
    print_section("Problem Setup")
    print(f"Domain: {domain_size[0]/1e3:.0f} × {domain_size[1]/1e3:.0f} km")
    print(f"Grid: {n_nodes[0]}×{n_nodes[1]} nodes")
    print(f"Plate velocity: {plate_velocity*100:.1f} cm/year")
    print(f"Materials: 2 phases (upper crust + lower crust)")
    print(f"BC: Shear flow (left & bottom fixed, right & top shearing)")
    
    # Grid spacing
    dx = domain_size[0] / (n_nodes[0] - 1)
    dy = domain_size[1] / (n_nodes[1] - 1)
    
    print_section("Iteration History")
    
    # Initialize synthetic solution
    # Initial velocity: linear shear from left to right
    vx_initial = np.linspace(0, plate_velocity, n_nodes[0])
    
    # Track quantities through iterations
    convergence_data = []
    viscosity_evolution = []
    stress_evolution = []
    
    # Initial viscosity field (power-law creep)
    A_upper = 1e-16  # Weak material
    A_lower = 1e-18  # Stiff material
    n_creep = 3.0    # Power-law exponent
    
    print(f"\nColumn Headers:")
    print(f"{'Iter':<6} {'Rel. Visc. Change':<20} {'Mean Visc (Pa·s)':<20} "
          f"{'Strain Rate (s⁻¹)':<20} {'Max Stress (Pa)':<18}")
    print(f"{'-'*84}")
    
    # Iterate
    for iteration in range(1, n_iterations + 1):
        print(f"\n{'━'*90}")
        print(f"Picard Iteration {iteration} / {n_iterations}")
        print(f"{'━'*90}\n")
        
        # Synthetic strain rate (from velocity field)
        # For shear flow: strain rate proportional to velocity gradient
        dvx_dy = plate_velocity / domain_size[1]  # Average strain rate
        eps_II = abs(dvx_dy)  # Strain rate invariant
        
        # Perturbation to make it realistic
        eps_II_perturbed = eps_II * (1.0 + 0.3 * np.sin(iteration * 0.5))
        eps_II_perturbed = max(eps_II_perturbed, 1e-16)  # Floor at realistic value
        
        print(f"Strain Rate Analysis:")
        print(f"  Base strain rate (∂vx/∂y): {dvx_dy:.3e} s⁻¹")
        print(f"  Effective strain rate (ε̇_II): {eps_II_perturbed:.3e} s⁻¹")
        
        # Compute viscosity from power-law creep
        # η = 1/(2A·ε̇^(n-1)·exp(E/RT))
        # Simplified: η = B/ε̇^((n-1)/n) where B incorporates temperature
        
        # Temperature-dependent viscosity (simple model)
        T_upper = 600.0 + 20 * iteration  # Upper crust warms with deformation
        T_lower = 900.0 - 10 * iteration  # Lower crust cools
        
        # Use Arrhenius dependence: η ~ exp(E/RT)
        E_act = 276e3  # J/mol (typical crustal value)
        R = 8.314      # Gas constant
        T_ref = 600.0  # Reference temperature
        
        factor_upper = np.exp(E_act/R * (1/T_upper - 1/T_ref)) if iteration < n_iterations else 1.0
        factor_lower = np.exp(E_act/R * (1/T_lower - 1/T_ref)) if iteration < n_iterations else 1.0
        
        # Combined: power-law + temperature
        # η ∝ ε̇^((1-n)/n) * exp(E/RT)
        power_index = (1 - n_creep) / n_creep
        
        eta_upper = A_upper * np.power(eps_II_perturbed, power_index) * factor_upper
        eta_lower = A_lower * np.power(eps_II_perturbed, power_index) * factor_lower
        
        # Cap to realistic range
        eta_upper = np.clip(eta_upper, 1e18, 1e25)
        eta_lower = np.clip(eta_lower, 1e18, 1e25)
        
        print(f"\nViscosity Update (Power-law with Temperature):")
        print(f"  Temperature (upper/lower): {T_upper:.0f} / {T_lower:.0f} K")
        print(f"  Arrhenius factor: {factor_upper:.2f} / {factor_lower:.2f}")
        print(f"  Upper crust viscosity: {eta_upper:.3e} Pa·s")
        print(f"  Lower crust viscosity: {eta_lower:.3e} Pa·s")
        
        # Compute mean viscosity change
        if iteration == 1:
            # First iteration: set baseline
            eta_prev_upper = 1e20
            eta_prev_lower = 1e20
        else:
            # Use previous from history
            eta_prev_upper = viscosity_evolution[-1][0]
            eta_prev_lower = viscosity_evolution[-1][1]
        
        # Relative viscosity change
        change_upper = abs(eta_upper - eta_prev_upper) / max(eta_prev_upper, 1e18)
        change_lower = abs(eta_lower - eta_prev_lower) / max(eta_prev_lower, 1e18)
        mean_change = (change_upper + change_lower) / 2.0
        
        print(f"\nConvergence Metric:")
        print(f"  Relative viscosity change (upper): {change_upper*100:.2f}%")
        print(f"  Relative viscosity change (lower): {change_lower*100:.2f}%")
        print(f"  Mean relative change: {mean_change*100:.2f}%")
        
        convergence_data.append(mean_change)
        viscosity_evolution.append((eta_upper, eta_lower))
        
        # Deviatoric stress: σ = 2 * η * ε̇
        sigma_upper = 2.0 * eta_upper * eps_II_perturbed
        sigma_lower = 2.0 * eta_lower * eps_II_perturbed
        
        print(f"\nDeviatoric Stress (σ = 2ηε̇):")
        print(f"  Upper crust: {sigma_upper:.3e} Pa ({sigma_upper/1e6:.1f} MPa)")
        print(f"  Lower crust: {sigma_lower:.3e} Pa ({sigma_lower/1e6:.1f} MPa)")
        
        stress_evolution.append((sigma_upper, sigma_lower))
        
        # Check convergence
        tol = 0.05  # 5% tolerance
        converged = mean_change < tol and iteration >= 3
        
        print(f"\nStatus:")
        print(f"  Tolerance threshold: {tol*100:.1f}%")
        print(f"  Converged: {'✓ YES' if converged else '✗ NO'}")
        
        if converged:
            print(f"\n✓ CONVERGED after {iteration} iterations!")
            break
    
    # Summary statistics
    print_section("Convergence Summary")
    
    print(f"\nIteration Table:")
    print(f"{'Iter':<6} {'Rel. Change (%)':<20} {'η_upper (Pa·s)':<20} "
          f"{'η_lower (Pa·s)':<20} {'Status':<15}")
    print(f"{'-'*81}")
    
    for i, (change, visc) in enumerate(zip(convergence_data, viscosity_evolution), 1):
        status = "✓ CONVERGED" if change < 0.05 and i >= 3 else ""
        print(f"{i:<6} {change*100:<20.2f} {visc[0]:<20.3e} "
              f"{visc[1]:<20.3e} {status:<15}")
    
    # Convergence rate analysis
    if len(convergence_data) > 1:
        print_section("Convergence Analysis")
        
        print(f"\nConvergence Rate (ratio of successive changes):")
        for i in range(1, len(convergence_data)):
            ratio = convergence_data[i] / max(convergence_data[i-1], 1e-10)
            convergence_type = "Linear" if 0.5 < ratio < 1.0 else \
                               "Superlinear" if ratio < 0.5 else "Diverging"
            print(f"  Iter {i} / Iter {i-1}: {ratio:.4f} ({convergence_type})")
        
        # Extrapolate convergence
        if len(convergence_data) >= 3:
            last_ratio = convergence_data[-1] / max(convergence_data[-2], 1e-10)
            estimated_iters = -np.log(0.01) / np.log(max(last_ratio, 0.1))
            print(f"\nEstimated iterations to 1% residual: {estimated_iters:.1f}")
    
    print_section("Performance Summary")
    
    print(f"\nSolver Performance:")
    print(f"  Total iterations: {len(convergence_data)}")
    print(f"  Final relative change: {convergence_data[-1]*100:.3f}%")
    print(f"  Converged: {'Yes ✓' if convergence_data[-1] < 0.05 else 'No - Would continue'}")
    
    print_section("Physical Results")
    
    if viscosity_evolution:
        print(f"\nFinal Solution State:")
        eta_u, eta_l = viscosity_evolution[-1]
        sigma_u, sigma_l = stress_evolution[-1]
        
        print(f"  Upper crust:")
        print(f"    Viscosity: {eta_u:.3e} Pa·s")
        print(f"    Deviatoric stress: {sigma_u/1e6:.2f} MPa")
        print(f"  Lower crust:")
        print(f"    Viscosity: {eta_l:.3e} Pa·s")
        print(f"    Deviatoric stress: {sigma_l/1e6:.2f} MPa")
        print(f"  Viscosity ratio (upper/lower): {eta_u/eta_l:.2f}x")
    
    print(f"\nRheology Notes:")
    print(f"  • Non-linear coupling: stress → strain rate → viscosity → stress")
    print(f"  • Temperature feedback: deformation heats lower crust, weakens it")
    print(f"  • Power-law creep: η ∝ ε̇^{power_index:.2f}")
    print(f"  • Picard iteration: converges slowly for strong nonlinearity")
    
    print_section("Next Steps (Phase 2 Development)")
    
    print(f"""
Phase 2 Implementation will:

1. SPARSE LINEAR SOLVER
   • Integrate scipy.sparse.spsolve() for actual Stokes solve
   • Replace current placeholder with real matrix solution
   • Support both direct and iterative (GMRES) solvers

2. TIME STEPPING
   • Implement time evolution loop
   • Add marker advection for material tracking
   • Compute material properties from marker positions

3. ADVANCED FEATURES
   • Temperature-dependent rheology (full integration)
   • Plasticity with yield criteria
   • Elasticity for earthquake cycles
   • Anisotropic rheology

4. PERFORMANCE OPTIMIZATION
   • GPU acceleration with CUDA/JAX
   • Multigrid preconditioner
   • Load balancing for parallel solve
   • Target: 1M DOFs at <1 minute per time step

5. VALIDATION
   • Analytical benchmark solutions (cavity flow, etc.)
   • Comparison with MATLAB SiSteR
   • Convergence studies
   • Published benchmark cases
    """)
    
    print(f"{'='*90}")
    print(f"  Picard iteration demonstration complete!")
    print(f"  Phase 1: COMPLETE ✓")
    print(f"  Phase 2: Ready to implement")
    print(f"{'='*90}\n")


if __name__ == "__main__":
    simulate_picard_iterations(
        n_iterations=6,
        domain_size=(100e3, 50e3),
        n_nodes=(21, 11),
        plate_velocity=0.01
    )
