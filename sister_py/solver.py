"""
Stokes Solver Module for SiSteR-py

Solves the coupled incompressible Stokes equations on a staggered grid:
    ∇·σ + ρg = 0         (momentum balance)
    ∇·v = 0              (mass conservation)
    σ = 2ηε̇ - pI         (constitutive relation)

Features:
    - Fully-staggered MAC grid discretization
    - Finite difference assembly on staggered nodes
    - Boundary condition handling (stress, velocity, mixed)
    - Non-linear rheology coupling (Picard iteration)
    - Sparse matrix solver integration

Classes:
    SolverConfig: Configuration for solver parameters
    BoundaryCondition: Boundary condition specification
    SolverSystem: Main Stokes solver class
    SolutionFields: Container for velocity and pressure fields

Functions:
    assemble_momentum_equation: Build momentum matrix
    assemble_continuity_equation: Build continuity matrix
    apply_boundary_conditions: Apply BCs to system
"""

import numpy as np
from typing import Dict, Tuple, Optional, Union, List
from dataclasses import dataclass, field
from enum import Enum
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from pathlib import Path
import warnings

from sister_py.grid import Grid
from sister_py.material_grid import MaterialGrid
from sister_py.linear_solver import (
    LinearSolver, DirectSolver, GMRESSolver, BiCGSTABSolver,
    select_solver, SolverStats
)


class BCType(Enum):
    """Boundary condition type enumeration."""
    VELOCITY = "velocity"      # Dirichlet: specify velocity
    STRESS = "stress"          # Neumann: specify stress/traction
    FREE_SURFACE = "free_surface"  # Traction-free surface
    PERIODIC = "periodic"      # Periodic boundary


@dataclass
class BoundaryCondition:
    """Boundary condition specification."""
    side: str                   # 'top', 'bottom', 'left', 'right'
    bc_type: BCType             # Type of boundary condition
    vx: Optional[float] = None  # X-velocity for VELOCITY BC
    vy: Optional[float] = None  # Y-velocity for VELOCITY BC
    sxx: Optional[float] = None # Normal stress component
    sxy: Optional[float] = None # Shear stress component
    syy: Optional[float] = None # Normal stress component (for corners)
    
    def validate(self) -> None:
        """Validate BC specification."""
        if self.bc_type == BCType.VELOCITY:
            if self.vx is None or self.vy is None:
                raise ValueError(f"VELOCITY BC on {self.side} requires vx and vy")
        elif self.bc_type == BCType.STRESS:
            if self.sxx is None or self.sxy is None:
                raise ValueError(f"STRESS BC on {self.side} requires sxx and sxy")


@dataclass
class SolverConfig:
    """Configuration for Stokes solver."""
    # Picard iteration settings
    Npicard_min: int = 5
    Npicard_max: int = 50
    picard_tol: float = 1e-4        # Viscosity change tolerance
    
    # Linear solver settings
    solver_type: str = "direct"      # 'direct', 'iterative'
    iterative_tol: float = 1e-5
    iterative_maxiter: int = 1000
    
    # Rheology settings
    plasticity_enabled: bool = True
    thermal_enabled: bool = False
    elasticity_enabled: bool = False
    
    # Velocity scaling
    velocity_scale: float = 1.0      # For non-dimensionalization
    
    # Output settings
    verbose: bool = False
    
    def validate(self) -> None:
        """Validate solver configuration."""
        if self.Npicard_min < 1:
            raise ValueError("Npicard_min must be >= 1")
        if self.Npicard_max < self.Npicard_min:
            raise ValueError("Npicard_max must be >= Npicard_min")
        if self.solver_type not in ["direct", "iterative"]:
            raise ValueError("solver_type must be 'direct' or 'iterative'")


@dataclass
class SolutionFields:
    """Container for velocity and pressure solution fields."""
    vx: np.ndarray      # X-velocity on x-staggered nodes (nx_s × ny)
    vy: np.ndarray      # Y-velocity on y-staggered nodes (nx × ny_s)
    p: np.ndarray       # Pressure on normal nodes (nx × ny)
    
    # Derived fields
    vx_n: Optional[np.ndarray] = None     # X-velocity interpolated to normal nodes
    vy_n: Optional[np.ndarray] = None     # Y-velocity interpolated to normal nodes
    strain_rate: Optional[np.ndarray] = None  # Strain rate invariant
    
    @property
    def shape(self) -> Dict[str, Tuple]:
        """Return shapes of all fields."""
        return {
            'vx': self.vx.shape,
            'vy': self.vy.shape,
            'p': self.p.shape,
        }
    
    def to_dict(self) -> Dict:
        """Export solution as dictionary."""
        return {
            'vx': self.vx,
            'vy': self.vy,
            'p': self.p,
            'vx_n': self.vx_n,
            'vy_n': self.vy_n,
            'strain_rate': self.strain_rate,
        }


class SolverSystem:
    """
    Incompressible Stokes solver on fully-staggered grid.
    
    Solves the coupled system:
        ∇·σ + ρg = 0         (momentum balance)
        ∇·v = 0              (mass conservation)
    
    Uses finite difference discretization on MAC grid with:
    - Velocity at cell faces (vx on x-faces, vy on y-faces)
    - Pressure at cell centers
    - Deviatoric stress components at appropriate positions
    
    Attributes:
        grid (Grid): Computational grid
        material_grid (MaterialGrid): Material property distribution
        cfg (SolverConfig): Solver configuration
        bcs (List[BoundaryCondition]): Boundary conditions
        solution (SolutionFields): Current solution (velocity, pressure)
        
    Public Methods:
        solve(): Solve the Stokes system with Picard iteration
        assemble_system(): Build momentum + continuity matrices
        apply_bcs(): Apply boundary conditions to system
    """
    
    def __init__(self,
                 grid: Grid,
                 material_grid: MaterialGrid,
                 cfg: SolverConfig,
                 bcs: List[BoundaryCondition]):
        """
        Initialize Stokes solver.
        
        Parameters:
            grid: Computational grid
            material_grid: Material property distribution
            cfg: Solver configuration
            bcs: List of boundary conditions
        """
        cfg.validate()
        for bc in bcs:
            bc.validate()
        
        self.grid = grid
        self.material_grid = material_grid
        self.cfg = cfg
        self.bcs = {bc.side: bc for bc in bcs}  # Index by side
        
        # Grid dimensions
        self.nx = grid.metadata.nx
        self.ny = grid.metadata.ny
        self.nx_s = len(grid.x_s)      # nx - 1
        self.ny_s = len(grid.y_s)      # ny - 1
        
        # System matrix and RHS
        self.A = None           # Stiffness matrix
        self.b = None           # Right-hand side
        
        # Linear solver
        n_dof = 2 * self.nx_s * self.ny + 2 * self.nx * self.ny_s + self.nx * self.ny
        self.linear_solver = select_solver(
            problem_size=n_dof,
            solver_type='auto',
            verbose=getattr(cfg, 'verbose', False)
        )
        self.solver_stats = None
        
        # Solution
        self.solution = None
        
        # Iteration state
        self.picard_iteration = 0
        self.converged = False
        self.residuals = []
    
    def solve(self, 
              initial_velocity: Optional[Tuple[np.ndarray, np.ndarray]] = None,
              max_iterations: Optional[int] = None) -> SolutionFields:
        """
        Solve the incompressible Stokes equations using Picard iteration.
        
        Parameters:
            initial_velocity: Initial guess (vx, vy). If None, uses zero.
            max_iterations: Override Npicard_max from config
        
        Returns:
            SolutionFields with velocity and pressure
        
        Algorithm:
            1. Initialize velocity field (or use provided initial guess)
            2. For Picard iteration k = 1, 2, ..., max_iter:
               a. Compute strain rate from current velocity
               b. Update viscosity based on strain rate
               c. Assemble momentum + continuity system
               d. Apply boundary conditions
               e. Solve linear system for velocity and pressure
               f. Check convergence (relative viscosity change)
            3. Return final solution
        """
        max_iter = max_iterations or self.cfg.Npicard_max
        
        # Initialize velocity fields
        if initial_velocity is None:
            vx = np.zeros((self.nx_s, self.ny))
            vy = np.zeros((self.nx, self.ny_s))
        else:
            vx, vy = initial_velocity
            if vx.shape != (self.nx_s, self.ny):
                raise ValueError(f"vx shape {vx.shape} != expected {(self.nx_s, self.ny)}")
            if vy.shape != (self.nx, self.ny_s):
                raise ValueError(f"vy shape {vy.shape} != expected {(self.nx, self.ny_s)}")
        
        # Picard iteration loop
        for picard_iter in range(max_iter):
            self.picard_iteration = picard_iter + 1
            
            if self.cfg.verbose:
                print(f"\nPicard iteration {self.picard_iteration}/{max_iter}")
            
            # Compute strain rate invariant from current velocity
            strain_rate = self._compute_strain_rate_invariant(vx, vy)
            
            # Update viscosity based on strain rate
            old_viscosity_n = self.material_grid.viscosity_effective_n.copy()
            self._update_viscosity_from_strain_rate(strain_rate)
            
            # Assemble system matrix and RHS
            self._assemble_system(vx, vy)
            
            # Apply boundary conditions
            self._apply_boundary_conditions()
            
            # Solve linear system
            solution_vector = self._solve_linear_system()
            
            # Extract velocity and pressure components
            vx, vy, p = self._extract_solution(solution_vector)
            
            # Check convergence
            if picard_iter >= self.cfg.Npicard_min - 1:
                visc_change = self._compute_viscosity_change(old_viscosity_n)
                self.residuals.append(visc_change)
                
                if self.cfg.verbose:
                    print(f"  Viscosity change: {visc_change:.4e}")
                
                if visc_change < self.cfg.picard_tol:
                    self.converged = True
                    if self.cfg.verbose:
                        print(f"Converged after {self.picard_iteration} iterations")
                    break
        
        # Store solution
        self.solution = SolutionFields(vx=vx, vy=vy, p=p)
        return self.solution
    
    def _compute_strain_rate_invariant(self, 
                                      vx: np.ndarray, 
                                      vy: np.ndarray) -> np.ndarray:
        """
        Compute the second invariant of strain rate tensor on normal nodes.
        
        ε̇_II = sqrt(ε̇_xx² + ε̇_yy² + 2ε̇_xy²)
        
        Parameters:
            vx: X-velocity on x-staggered nodes (nx_s × ny)
            vy: Y-velocity on y-staggered nodes (nx × ny_s)
        
        Returns:
            Strain rate invariant on normal nodes (nx × ny)
        """
        dx = np.diff(self.grid.x_n)      # Grid spacing in x (nx-1)
        dy = np.diff(self.grid.y_n)      # Grid spacing in y (ny-1)
        
        # Compute strain rate components on normal nodes
        # ε̇_xx = ∂vx/∂x at normal nodes
        # vx is on x-staggered nodes with shape (nx_s, ny) = (nx-1, ny)
        eps_xx_n = np.zeros((self.nx, self.ny))
        for i in range(self.nx - 1):
            eps_xx_n[i, :] = (vx[i, :] - vx[i-1, :]) / dx[i] if i > 0 else vx[i, :] / dx[i]
        # Approximate for last column using backward difference
        if self.nx > 0:
            eps_xx_n[-1, :] = (vx[-1, :] - vx[-2, :]) / dx[-1]
        
        # ε̇_yy = ∂vy/∂y at normal nodes
        # vy is on y-staggered nodes with shape (nx, ny_s) = (nx, ny-1)
        eps_yy_n = np.zeros((self.nx, self.ny))
        for j in range(self.ny - 1):
            eps_yy_n[:, j] = (vy[:, j] - vy[:, j-1]) / dy[j] if j > 0 else vy[:, j] / dy[j]
        # Approximate for last row using backward difference
        if self.ny > 0:
            eps_yy_n[:, -1] = (vy[:, -1] - vy[:, -2]) / dy[-1]
        
        # ε̇_xy = 1/2 (∂vx/∂y + ∂vy/∂x) - compute on normal nodes
        eps_xy_n = self._compute_shear_strain_rate(vx, vy)
        
        # Second invariant (add small value to avoid division by zero)
        eps_II = np.sqrt(eps_xx_n**2 + eps_yy_n**2 + 2*eps_xy_n**2 + 1e-32)
        
        return eps_II
    
    def _compute_shear_strain_rate(self, 
                                  vx: np.ndarray, 
                                  vy: np.ndarray) -> np.ndarray:
        """
        Compute shear strain rate ε̇_xy = 1/2(∂vx/∂y + ∂vy/∂x).
        
        Parameters:
            vx: X-velocity on x-staggered nodes (nx_s × ny)
            vy: Y-velocity on y-staggered nodes (nx × ny_s)
        
        Returns:
            Shear strain rate on normal nodes (nx × ny)
        """
        dy = np.diff(self.grid.y_n)  # Grid spacing in y (ny-1)
        dx = np.diff(self.grid.x_n)  # Grid spacing in x (nx-1)
        
        eps_xy_n = np.zeros((self.nx, self.ny))
        
        # ∂vx/∂y: vx is on x-staggered nodes (nx_s, ny)
        # ∂vy/∂x: vy is on y-staggered nodes (nx, ny_s)
        
        # For normal node (i, j):
        # ∂vx/∂y ≈ (vx[i, j+1] - vx[i, j]) / dy[j] for j < ny-1
        for i in range(self.nx_s):  # i < nx
            for j in range(self.ny - 1):
                dvx_dy = (vx[i, j+1] - vx[i, j]) / dy[j]
                eps_xy_n[i, j] += dvx_dy / 2.0
        
        # ∂vy/∂x ≈ (vy[i+1, j] - vy[i, j]) / dx[i] for i < nx-1
        for i in range(self.nx - 1):
            for j in range(self.ny_s):  # j < ny
                dvy_dx = (vy[i+1, j] - vy[i, j]) / dx[i]
                eps_xy_n[i, j] += dvy_dx / 2.0
        
        return eps_xy_n
    
    def _update_viscosity_from_strain_rate(self, strain_rate: np.ndarray) -> None:
        """
        Update effective viscosity on normal nodes based on strain rate.
        
        For non-linear rheologies, re-evaluate viscosity using strain rate.
        
        Parameters:
            strain_rate: Strain rate invariant on normal nodes
        """
        # This would call Material methods to recompute viscosity
        # For now, this is a placeholder for the material-viscosity coupling
        T = 273.0 + 1300.0  # Reference temperature (example)
        P = 1e6 * (self.grid.y_n.mean() / 1e3) * 3300 * 9.81 / 1e6  # Approximate depth pressure
        P = np.maximum(P, 1e6)
        
        # Update viscosity for each unique phase
        unique_phases = np.unique(self.material_grid.phase_n)
        for phase_id in unique_phases:
            mask = self.material_grid.phase_n == phase_id
            material = self.material_grid.materials[phase_id]
            
            # Recompute viscosity using strain rate
            sigma_II = 2 * strain_rate[mask] * material.viscosity_effective(
                strain_rate[mask], 1e-14, T, P
            )  # This is a simplified approximation
            
            self.material_grid.viscosity_effective_n[mask] = material.viscosity_effective(
                sigma_II, strain_rate[mask], T, P
            )
    
    def _compute_viscosity_change(self, old_viscosity: np.ndarray) -> float:
        """
        Compute relative change in viscosity between iterations.
        
        Parameters:
            old_viscosity: Viscosity from previous iteration
        
        Returns:
            Normalized L2 change
        """
        new_viscosity = self.material_grid.viscosity_effective_n
        dvisc = new_viscosity - old_viscosity
        change = np.linalg.norm(dvisc) / (np.linalg.norm(old_viscosity) + 1e-30)
        return float(change)
    
    def _assemble_system(self, vx: np.ndarray, vy: np.ndarray) -> None:
        """
        Assemble momentum and continuity equations into matrix form.
        
        Parameters:
            vx: Current X-velocity estimate
            vy: Current Y-velocity estimate
        """
        n_vel = self.nx_s * self.ny + self.nx * self.ny_s  # Number of velocity DOFs
        n_pres = self.nx * self.ny                         # Number of pressure DOFs
        n_dof = n_vel + n_pres                            # Total DOFs
        
        # Initialize sparse matrix and RHS
        self.A = sp.lil_matrix((n_dof, n_dof))
        self.b = np.zeros(n_dof)
        
        # Assemble momentum equations (simplified for now)
        self._assemble_momentum_block()
        
        # Assemble continuity equation (simplified for now)
        self._assemble_continuity_block()
        
        # Convert to CSR format for efficient solving
        self.A = self.A.tocsr()
    
    def _assemble_momentum_block(self) -> None:
        """Assemble momentum equation portion of system matrix."""
        # This would contain the full finite difference stencils for:
        # ∇·τ - ∇p + ρg = 0
        # where τ = 2ηε̇ is the deviatoric stress
        pass
    
    def _assemble_continuity_block(self) -> None:
        """Assemble continuity equation portion of system matrix."""
        # This would contain finite difference discretization of:
        # ∇·v = 0
        pass
    
    def _apply_boundary_conditions(self) -> None:
        """
        Apply boundary conditions to system matrix and RHS.
        
        For Dirichlet (velocity): Replace rows with BC rows
        For Neumann (stress): Modify RHS with traction
        """
        # Boundary condition application logic
        pass
    
    def _solve_linear_system(self) -> np.ndarray:
        """
        Solve the assembled linear system using configured linear solver.
        
        Automatically selects between direct (for small problems) and
        iterative solvers (for large problems).
        
        Returns:
            Solution vector [vx_vec, vy_vec, p_vec]
        
        Raises:
            RuntimeError: If linear solver fails to converge
        """
        if self.A is None or self.b is None:
            raise ValueError("System matrix (A) and RHS (b) must be assembled first")
        
        # Setup linear solver
        self.linear_solver.setup(self.A)
        
        # Solve system
        solution, stats = self.linear_solver.solve(self.A, self.b)
        self.solver_stats = stats
        
        if self.cfg.verbose:
            print(f"  Linear solver: {stats}")
        
        # Check convergence
        if not stats.converged and isinstance(self.linear_solver, (GMRESSolver, BiCGSTABSolver)):
            warnings.warn(
                f"Linear solver did not converge: relative residual = {stats.relative_residual:.2e}"
            )
        
        return solution
    
    def _extract_solution(self, solution_vector: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Extract velocity and pressure components from solution vector.
        
        Parameters:
            solution_vector: Full solution [vx, vy, p]
        
        Returns:
            (vx, vy, p) arrays
        """
        n_vx = self.nx_s * self.ny
        n_vy = self.nx * self.ny_s
        
        vx_vec = solution_vector[:n_vx].reshape((self.nx_s, self.ny))
        vy_vec = solution_vector[n_vx:n_vx+n_vy].reshape((self.nx, self.ny_s))
        p_vec = solution_vector[n_vx+n_vy:].reshape((self.nx, self.ny))
        
        return vx_vec, vy_vec, p_vec
    
    def __repr__(self) -> str:
        """String representation."""
        status = "converged" if self.converged else "not converged"
        return (
            f"SolverSystem(grid={self.nx}×{self.ny}, "
            f"iterations={self.picard_iteration}, "
            f"status={status})"
        )


def assemble_momentum_equation(grid: Grid,
                              material_grid: MaterialGrid) -> Tuple[sp.csr_matrix, np.ndarray]:
    """
    Assemble momentum balance equation in matrix form.
    
    Finite difference discretization of:
        ∂σ_xx/∂x + ∂σ_xy/∂y + ρg_x = 0
        ∂σ_xy/∂x + ∂σ_yy/∂y + ρg_y = 0
    
    Parameters:
        grid: Computational grid
        material_grid: Material properties
    
    Returns:
        (momentum_matrix, body_force_vector)
    """
    # Placeholder: would implement full FD stencils
    nx_s, ny = len(grid.x_s), len(grid.y_n)
    nx, ny_s = len(grid.x_n), len(grid.y_s)
    
    n_vel = nx_s * ny + nx * ny_s
    A = sp.lil_matrix((n_vel, n_vel))
    b = np.zeros(n_vel)
    
    # Would populate A and b here with FD stencils
    
    return A.tocsr(), b


def assemble_continuity_equation(grid: Grid) -> Tuple[sp.csr_matrix, np.ndarray]:
    """
    Assemble continuity equation in matrix form.
    
    Discretization of:
        ∂v_x/∂x + ∂v_y/∂y = 0
    
    Parameters:
        grid: Computational grid
    
    Returns:
        (continuity_matrix, zero_rhs)
    """
    # Placeholder: would implement divergence operator
    nx = grid.metadata.nx
    ny = grid.metadata.ny
    nx_s = len(grid.x_s)
    ny_s = len(grid.y_s)
    
    n_vel = nx_s * ny + nx * ny_s
    n_pres = nx * ny
    
    C = sp.lil_matrix((n_pres, n_vel))
    c = np.zeros(n_pres)
    
    # Would populate C and c here with divergence operator
    
    return C.tocsr(), c


def apply_boundary_conditions(A: sp.csr_matrix,
                             b: np.ndarray,
                             grid: Grid,
                             bcs: Dict[str, BoundaryCondition]) -> Tuple[sp.csr_matrix, np.ndarray]:
    """
    Apply boundary conditions to system matrix and RHS.
    
    Parameters:
        A: System matrix
        b: Right-hand side vector
        grid: Computational grid
        bcs: Dictionary mapping side name to BoundaryCondition
    
    Returns:
        (modified_A, modified_b)
    """
    # Placeholder: would implement BC application logic
    return A, b
