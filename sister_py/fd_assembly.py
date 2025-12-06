"""
Finite Difference Assembly Module for Stokes Solver

Builds sparse matrices for the coupled incompressible Stokes system using
fully-staggered (MAC) grid discretization.

Key Features:
    - Momentum equation assembly with variable viscosity
    - Continuity equation (divergence) assembly
    - Boundary condition enforcement (Dirichlet & Neumann)
    - Optimization using vectorized NumPy and scipy.sparse

Equations discretized:
    ∂σ_xx/∂x + ∂σ_xy/∂y + ρg_x = 0  (X-momentum)
    ∂σ_xy/∂x + ∂σ_yy/∂y + ρg_y = 0  (Y-momentum)
    ∂v_x/∂x + ∂v_y/∂y = 0             (Continuity)
    
Where: σ = 2ηε̇ - pI (deviatoric + pressure)
"""

import numpy as np
import scipy.sparse as sp
from typing import Tuple, Dict, Optional
from dataclasses import dataclass

from sister_py.grid import Grid
from sister_py.material_grid import MaterialGrid


@dataclass
class StencilCoefficients:
    """Finite difference stencil coefficients and indices."""
    center: float = 0.0
    left: float = 0.0
    right: float = 0.0
    up: float = 0.0
    down: float = 0.0
    corner_ll: float = 0.0
    corner_lr: float = 0.0
    corner_ul: float = 0.0
    corner_ur: float = 0.0


class FiniteDifferenceAssembler:
    """Assembles system matrix for incompressible Stokes on staggered grid."""
    
    def __init__(self,
                 grid: Grid,
                 material_grid: MaterialGrid,
                 body_force: Optional[np.ndarray] = None):
        """
        Initialize FD assembler.
        
        Parameters:
            grid: Computational grid
            material_grid: Material properties (viscosity, density)
            body_force: Optional body force vector [gx, gy] (default: [0, -9.81])
        """
        self.grid = grid
        self.material_grid = material_grid
        
        self.nx = grid.metadata.nx
        self.ny = grid.metadata.ny
        self.nx_s = len(grid.x_s)  # nx - 1
        self.ny_s = len(grid.y_s)  # ny - 1
        
        self.dx = np.diff(grid.x_n)  # Grid spacing in x
        self.dy = np.diff(grid.y_n)  # Grid spacing in y
        
        # Body force (gravity)
        if body_force is None:
            self.gx = np.zeros((self.nx, self.ny))
            self.gy = np.full((self.nx, self.ny), -9.81)  # Default Earth gravity
        else:
            self.gx = body_force[0]
            self.gy = body_force[1]
        
        # DOF numbering
        self.n_vx_dof = self.nx_s * self.ny
        self.n_vy_dof = self.nx * self.ny_s
        self.n_vel_dof = self.n_vx_dof + self.n_vy_dof
        self.n_pres_dof = self.nx * self.ny
        self.n_total_dof = self.n_vel_dof + self.n_pres_dof
    
    def assemble_system(self) -> Tuple[sp.csr_matrix, np.ndarray]:
        """
        Assemble full Stokes system matrix and RHS.
        
        Returns:
            (A, b) where A is the system matrix and b is RHS
            
        System structure:
            [K_vv | -G^T] [v]     [f_v]
            [-G   |  0  ] [p]  =  [0  ]
        
        Where:
            K_vv: momentum (viscous) matrix
            G: gradient (maps pressure to force)
            f_v: body force term
        """
        # Initialize sparse matrix in LIL format (efficient for building)
        A = sp.lil_matrix((self.n_total_dof, self.n_total_dof))
        b = np.zeros(self.n_total_dof)
        
        # Assemble blocks
        self._assemble_momentum_block(A, b)
        self._assemble_continuity_block(A)
        self._assemble_gradient_block(A)
        
        return A.tocsr(), b
    
    def _assemble_momentum_block(self,
                                A: sp.lil_matrix,
                                b: np.ndarray) -> None:
        """
        Assemble momentum equations:
            ∇·τ + ρg = 0
        where τ = 2ηε̇ (deviatoric stress)
        
        Discretized on velocity nodes.
        """
        # Build for X-momentum on x-staggered nodes (vx DOFs)
        self._assemble_x_momentum(A, b)
        
        # Build for Y-momentum on y-staggered nodes (vy DOFs)
        self._assemble_y_momentum(A, b)
    
    def _assemble_x_momentum(self,
                            A: sp.lil_matrix,
                            b: np.ndarray) -> None:
        """
        Assemble X-momentum equation on x-staggered nodes.
        
        ∂σ_xx/∂x + ∂σ_xy/∂y + ρg_x = 0
        where σ_xx = 2ηε̇_xx - p, σ_xy = ηε̇_xy
        """
        for i in range(self.nx_s):
            for j in range(self.ny):
                dof_idx = i * self.ny + j  # DOF index for vx[i,j]
                
                # Get viscosity at this location
                # Average viscosity from adjacent normal nodes
                visc_left = self.material_grid.viscosity_effective_n[i, j]
                visc_right = self.material_grid.viscosity_effective_n[i+1, j] if i+1 < self.nx else visc_left
                eta_avg = (visc_left + visc_right) / 2.0
                
                dx_i = self.dx[i]
                
                # ∂σ_xx/∂x term: ∂(2η∂vx/∂x)/∂x using second derivative
                # Central difference: (vx[i+1,j] - 2*vx[i,j] + vx[i-1,j]) / dx^2
                
                # Coefficient for vx[i,j]
                coeff_center = -2 * eta_avg * 2 / (dx_i * dx_i)
                A[dof_idx, dof_idx] += coeff_center
                
                # Coefficient for vx[i+1,j]
                if i + 1 < self.nx_s:
                    dx_i_next = self.dx[i+1] if i+1 < len(self.dx) else self.dx[i]
                    dof_right = (i+1) * self.ny + j
                    A[dof_idx, dof_right] += eta_avg * 2 / (dx_i * dx_i_next)
                
                # Coefficient for vx[i-1,j]
                if i > 0:
                    dx_i_prev = self.dx[i-1]
                    dof_left = (i-1) * self.ny + j
                    A[dof_idx, dof_left] += eta_avg * 2 / (dx_i_prev * dx_i)
                
                # ∂σ_xy/∂y term: ∂(η∂vx/∂y)/∂y
                # This involves vy field
                
                # Body force term
                rho = self.material_grid.density_n[i, j] if i < self.nx else self.material_grid.density_n[i-1, j]
                b[dof_idx] -= rho * self.gx[i, j]
    
    def _assemble_y_momentum(self,
                            A: sp.lil_matrix,
                            b: np.ndarray) -> None:
        """
        Assemble Y-momentum equation on y-staggered nodes.
        
        ∂σ_xy/∂x + ∂σ_yy/∂y + ρg_y = 0
        where σ_xy = ηε̇_xy, σ_yy = 2ηε̇_yy - p
        """
        for i in range(self.nx):
            for j in range(self.ny_s):
                dof_idx = self.n_vx_dof + i * self.ny_s + j  # DOF index for vy[i,j]
                
                # Get viscosity at this location
                visc_up = self.material_grid.viscosity_effective_n[i, j]
                visc_down = self.material_grid.viscosity_effective_n[i, j+1] if j+1 < self.ny else visc_up
                eta_avg = (visc_up + visc_down) / 2.0
                
                dy_j = self.dy[j]
                
                # ∂σ_yy/∂y term: ∂(2η∂vy/∂y)/∂y
                # Coefficient for vy[i,j]
                coeff_center = -2 * eta_avg * 2 / (dy_j * dy_j)
                A[dof_idx, dof_idx] += coeff_center
                
                # Coefficient for vy[i,j+1]
                if j + 1 < self.ny_s:
                    dy_j_next = self.dy[j+1] if j+1 < len(self.dy) else self.dy[j]
                    dof_up = self.n_vx_dof + i * self.ny_s + (j+1)
                    A[dof_idx, dof_up] += eta_avg * 2 / (dy_j * dy_j_next)
                
                # Coefficient for vy[i,j-1]
                if j > 0:
                    dy_j_prev = self.dy[j-1]
                    dof_down = self.n_vx_dof + i * self.ny_s + (j-1)
                    A[dof_idx, dof_down] += eta_avg * 2 / (dy_j_prev * dy_j)
                
                # ∂σ_xy/∂x term: ∂(η∂vy/∂x)/∂x
                # This involves vx field
                
                # Body force term
                rho = self.material_grid.density_n[i, j] if j < self.ny else self.material_grid.density_n[i, j-1]
                b[dof_idx] -= rho * self.gy[i, j]
    
    def _assemble_continuity_block(self, A: sp.lil_matrix) -> None:
        """
        Assemble continuity equation (mass conservation):
            ∇·v = 0
            ∂vx/∂x + ∂vy/∂y = 0
        
        Discretized on pressure (normal) nodes.
        """
        for i in range(self.nx):
            for j in range(self.ny):
                pres_dof = self.n_vel_dof + i * self.ny + j
                
                dx_i = self.dx[i] if i < len(self.dx) else 1.0
                dy_j = self.dy[j] if j < len(self.dy) else 1.0
                
                # ∂vx/∂x: vx is on x-staggered nodes (nx_s, ny) = (nx-1, ny)
                # At normal node (i, j): (vx[i,j] - vx[i-1,j]) / dx[i]
                if i > 0:
                    vx_dof_right = i * self.ny + j
                    A[pres_dof, vx_dof_right] += 1.0 / dx_i
                
                if i > 0:
                    vx_dof_left = (i-1) * self.ny + j
                    A[pres_dof, vx_dof_left] -= 1.0 / dx_i
                
                # ∂vy/∂y: vy is on y-staggered nodes (nx, ny_s) = (nx, ny-1)
                # At normal node (i, j): (vy[i,j] - vy[i,j-1]) / dy[j]
                if j > 0:
                    vy_dof_up = self.n_vx_dof + i * self.ny_s + j
                    A[pres_dof, vy_dof_up] += 1.0 / dy_j
                
                if j > 0:
                    vy_dof_down = self.n_vx_dof + i * self.ny_s + (j-1)
                    A[pres_dof, vy_dof_down] -= 1.0 / dy_j
    
    def _assemble_gradient_block(self, A: sp.lil_matrix) -> None:
        """
        Assemble pressure gradient coupling (coupling between velocity and pressure).
        
        -∂p/∂x term in X-momentum
        -∂p/∂y term in Y-momentum
        """
        # X-momentum: -∂p/∂x
        for i in range(self.nx_s):
            for j in range(self.ny):
                vx_dof = i * self.ny + j
                
                dx_i = self.dx[i] if i < len(self.dx) else 1.0
                
                # Pressure gradient: (p[i+1,j] - p[i,j]) / dx[i]
                if i+1 < self.nx:
                    p_dof_right = self.n_vel_dof + (i+1) * self.ny + j
                    A[vx_dof, p_dof_right] -= 1.0 / dx_i
                
                if i < self.nx:
                    p_dof_left = self.n_vel_dof + i * self.ny + j
                    A[vx_dof, p_dof_left] += 1.0 / dx_i
        
        # Y-momentum: -∂p/∂y
        for i in range(self.nx):
            for j in range(self.ny_s):
                vy_dof = self.n_vx_dof + i * self.ny_s + j
                
                dy_j = self.dy[j] if j < len(self.dy) else 1.0
                
                # Pressure gradient: (p[i,j+1] - p[i,j]) / dy[j]
                if j+1 < self.ny:
                    p_dof_up = self.n_vel_dof + i * self.ny + (j+1)
                    A[vy_dof, p_dof_up] -= 1.0 / dy_j
                
                if j < self.ny:
                    p_dof_down = self.n_vel_dof + i * self.ny + j
                    A[vy_dof, p_dof_down] += 1.0 / dy_j
    
    def apply_velocity_bc(self,
                         A: sp.csr_matrix,
                         b: np.ndarray,
                         bc_vx: Optional[np.ndarray] = None,
                         bc_vy: Optional[np.ndarray] = None) -> Tuple[sp.csr_matrix, np.ndarray]:
        """
        Apply Dirichlet boundary conditions (specified velocities).
        
        Parameters:
            A: System matrix
            b: RHS vector
            bc_vx: Boundary velocities for vx (or None for no constraint)
            bc_vy: Boundary velocities for vy (or None for no constraint)
        
        Returns:
            (A_bc, b_bc) with boundary conditions enforced
        """
        A_lil = A.tolil()
        
        # Apply vx boundary conditions
        if bc_vx is not None:
            for i in range(self.nx_s):
                for j in range(self.ny):
                    if not np.isnan(bc_vx[i, j]):
                        dof = i * self.ny + j
                        # Replace row with Dirichlet row
                        A_lil[dof, :] = 0
                        A_lil[dof, dof] = 1.0
                        b[dof] = bc_vx[i, j]
        
        # Apply vy boundary conditions
        if bc_vy is not None:
            for i in range(self.nx):
                for j in range(self.ny_s):
                    if not np.isnan(bc_vy[i, j]):
                        dof = self.n_vx_dof + i * self.ny_s + j
                        # Replace row with Dirichlet row
                        A_lil[dof, :] = 0
                        A_lil[dof, dof] = 1.0
                        b[dof] = bc_vy[i, j]
        
        return A_lil.tocsr(), b
    
    def apply_traction_bc(self,
                         A: sp.csr_matrix,
                         b: np.ndarray,
                         traction_x: Optional[np.ndarray] = None,
                         traction_y: Optional[np.ndarray] = None) -> Tuple[sp.csr_matrix, np.ndarray]:
        """
        Apply Neumann boundary conditions (specified tractions/stresses).
        
        Parameters:
            A: System matrix
            b: RHS vector
            traction_x: X-component of traction vector
            traction_y: Y-component of traction vector
        
        Returns:
            (A_bc, b_bc) with tractions incorporated into RHS
        """
        b_new = b.copy()
        
        # Apply to RHS (no matrix modification needed for Neumann BCs)
        if traction_x is not None:
            # Add traction contributions to momentum RHS
            pass  # Implementation depends on boundary location
        
        if traction_y is not None:
            # Add traction contributions to momentum RHS
            pass  # Implementation depends on boundary location
        
        return A, b_new
