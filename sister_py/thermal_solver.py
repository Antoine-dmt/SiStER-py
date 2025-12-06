"""
Thermal Solver Module for SiSteR-py

Implements heat diffusion, advection-diffusion coupling, and thermal properties.

Classes:
    ThermalProperties: Material thermal properties (k, cp, rho)
    ThermalBoundaryCondition: Thermal boundary conditions (Dirichlet, Neumann, Robin)
    HeatDiffusionSolver: Steady-state and transient heat diffusion
    AdvectionDiffusionSolver: Coupled advection-diffusion solver
    ThermalModel: Complete thermal system coordinator
    ThermalOutput: Thermal field output management

Functions:
    compute_thermal_conductivity: Material-dependent conductivity
    compute_heat_capacity: Temperature-dependent heat capacity
    estimate_thermal_time_scale: Thermal diffusion time scale
    interpolate_temperature_to_markers: Map grid temperature to markers
"""

import numpy as np
from typing import Tuple, Optional, Dict, List, Callable
from dataclasses import dataclass
from abc import ABC, abstractmethod
from scipy import sparse
from scipy.sparse import linalg
import warnings


@dataclass
class ThermalProperties:
    """Thermal properties for materials."""
    k: float = 3.0  # Thermal conductivity (W/m/K)
    k_aniso_ratio: float = 1.0  # Anisotropy ratio for conductivity
    cp: float = 1000.0  # Heat capacity (J/kg/K)
    rho: float = 2800.0  # Density (kg/m³)
    alpha: float = 3e-5  # Thermal expansion coefficient (1/K)
    T_ref: float = 273.15  # Reference temperature (K)
    
    def __post_init__(self):
        """Validate properties."""
        if self.k <= 0:
            warnings.warn(f"Thermal conductivity k={self.k} should be positive")
        if self.cp <= 0:
            warnings.warn(f"Heat capacity cp={self.cp} should be positive")
        if self.rho <= 0:
            warnings.warn(f"Density rho={self.rho} should be positive")


class ThermalBoundaryCondition:
    """Boundary condition for thermal problems."""
    
    class BCType:
        """Boundary condition types."""
        DIRICHLET = 'dirichlet'  # Fixed temperature
        NEUMANN = 'neumann'      # Fixed heat flux
        ROBIN = 'robin'           # Mixed BC: -k*dT/dn = h(T - T_amb)
    
    def __init__(self,
                 boundary: str,
                 bc_type: str,
                 value: float = 0.0,
                 ambient_temp: float = 273.15,
                 h_coeff: float = 0.0):
        """
        Initialize thermal BC.
        
        Parameters:
            boundary: 'top', 'bottom', 'left', 'right'
            bc_type: 'dirichlet', 'neumann', or 'robin'
            value: For Dirichlet (T in K), for Neumann (heat flux in W/m²)
            ambient_temp: For Robin BC (K)
            h_coeff: For Robin BC (convection coefficient W/m²/K)
        """
        self.boundary = boundary
        self.bc_type = bc_type
        self.value = value
        self.ambient_temp = ambient_temp
        self.h_coeff = h_coeff
    
    def __repr__(self) -> str:
        return f"ThermalBC({self.boundary}, {self.bc_type}, value={self.value})"


@dataclass
class ThermalFieldData:
    """Thermal field data at a time step."""
    temperature: np.ndarray  # Temperature field (K)
    temperature_old: np.ndarray = None  # Previous time step
    heat_flux_x: np.ndarray = None  # Heat flux x-component (W/m²)
    heat_flux_y: np.ndarray = None  # Heat flux y-component (W/m²)
    heat_generation: np.ndarray = None  # Heat generation rate (W/m³)
    time: float = 0.0  # Current time (s)
    dt: float = 0.0  # Current time step (s)
    
    @property
    def temperature_gradient_magnitude(self) -> float:
        """RMS temperature gradient."""
        if self.heat_flux_x is None or self.heat_flux_y is None:
            return 0.0
        return np.sqrt(np.mean(self.heat_flux_x**2 + self.heat_flux_y**2))
    
    @property
    def temperature_extrema(self) -> Tuple[float, float]:
        """Min and max temperature."""
        return np.min(self.temperature), np.max(self.temperature)


class ThermalMaterialProperties:
    """Manage thermal properties on material grid."""
    
    def __init__(self, n_phases: int = 10):
        self.n_phases = n_phases
        self.properties = {}
        self.default_props = ThermalProperties()
    
    def set_properties(self, phase: int, props: ThermalProperties):
        """Set thermal properties for a phase."""
        self.properties[phase] = props
    
    def get_properties(self, phase: int) -> ThermalProperties:
        """Get thermal properties for a phase."""
        return self.properties.get(phase, self.default_props)
    
    def get_conductivity_field(self, phase_field: np.ndarray) -> np.ndarray:
        """Get conductivity values on grid from phase field."""
        k_field = np.zeros_like(phase_field, dtype=float)
        for phase in range(1, self.n_phases + 1):
            mask = phase_field == phase
            props = self.get_properties(phase)
            k_field[mask] = props.k
        return k_field
    
    def get_capacity_field(self, phase_field: np.ndarray) -> np.ndarray:
        """Get heat capacity values on grid from phase field."""
        cp_field = np.zeros_like(phase_field, dtype=float)
        for phase in range(1, self.n_phases + 1):
            mask = phase_field == phase
            props = self.get_properties(phase)
            cp_field[mask] = props.cp
        return cp_field
    
    def get_density_field(self, phase_field: np.ndarray) -> np.ndarray:
        """Get density values on grid from phase field."""
        rho_field = np.zeros_like(phase_field, dtype=float)
        for phase in range(1, self.n_phases + 1):
            mask = phase_field == phase
            props = self.get_properties(phase)
            rho_field[mask] = props.rho
        return rho_field


class HeatDiffusionSolver:
    """Solve steady-state and transient heat diffusion."""
    
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.n_iterations = 0
        self.residual_history = []
    
    def assemble_laplace_operator(self,
                                  k_field: np.ndarray,
                                  grid_x: np.ndarray,
                                  grid_y: np.ndarray,
                                  bc_conditions: Optional[List[ThermalBoundaryCondition]] = None) -> Tuple[sparse.csr_matrix, np.ndarray]:
        """
        Assemble Laplace operator: ∇·(k∇T) = 0
        
        Using 5-point stencil on regular grid:
        -k*(T_i-1,j - 2*T_i,j + T_i+1,j)/dx² - k*(T_i,j-1 - 2*T_i,j + T_i,j+1)/dy²
        
        Returns:
            (matrix, rhs_vector)
        """
        ny, nx = k_field.shape
        n_nodes = nx * ny
        
        # Grid spacing
        dx = grid_x[1] - grid_x[0] if len(grid_x) > 1 else 1.0
        dy = grid_y[1] - grid_y[0] if len(grid_y) > 1 else 1.0
        
        dx2 = dx * dx
        dy2 = dy * dy
        
        # Build matrix in COO format
        row = []
        col = []
        data = []
        rhs = np.zeros(n_nodes)
        
        for j in range(ny):
            for i in range(nx):
                idx = j * nx + i
                
                # Center coefficient
                diag_coeff = 2.0 * (k_field[j, i] / dx2 + k_field[j, i] / dy2)
                
                # x-direction (left)
                if i > 0:
                    k_west = 0.5 * (k_field[j, i] + k_field[j, i-1])
                    row.append(idx)
                    col.append(j * nx + (i-1))
                    data.append(-k_west / dx2)
                else:
                    # Neumann BC: dT/dx = 0 at boundary (insulated)
                    diag_coeff -= k_field[j, i] / dx2
                
                # x-direction (right)
                if i < nx - 1:
                    k_east = 0.5 * (k_field[j, i] + k_field[j, i+1])
                    row.append(idx)
                    col.append(j * nx + (i+1))
                    data.append(-k_east / dx2)
                else:
                    # Neumann BC at right boundary
                    diag_coeff -= k_field[j, i] / dx2
                
                # y-direction (bottom)
                if j > 0:
                    k_south = 0.5 * (k_field[j, i] + k_field[j-1, i])
                    row.append(idx)
                    col.append((j-1) * nx + i)
                    data.append(-k_south / dy2)
                else:
                    # Neumann BC: dT/dy = 0 at boundary
                    diag_coeff -= k_field[j, i] / dy2
                
                # y-direction (top)
                if j < ny - 1:
                    k_north = 0.5 * (k_field[j, i] + k_field[j+1, i])
                    row.append(idx)
                    col.append((j+1) * nx + i)
                    data.append(-k_north / dy2)
                else:
                    # Neumann BC at top boundary
                    diag_coeff -= k_field[j, i] / dy2
                
                # Diagonal
                row.append(idx)
                col.append(idx)
                data.append(diag_coeff)
        
        # Convert to CSR
        matrix = sparse.coo_matrix((data, (row, col)), shape=(n_nodes, n_nodes)).tocsr()
        
        return matrix, rhs
    
    def assemble_transient_operator(self,
                                   k_field: np.ndarray,
                                   cp_field: np.ndarray,
                                   rho_field: np.ndarray,
                                   grid_x: np.ndarray,
                                   grid_y: np.ndarray,
                                   dt: float) -> Tuple[sparse.csr_matrix, sparse.csr_matrix]:
        """
        Assemble operators for transient heat equation:
        ρ*cp*dT/dt - ∇·(k∇T) = Q
        
        Using backward Euler: (ρ*cp/dt + A) * T^{n+1} = ρ*cp/dt * T^n + Q
        
        Returns:
            (LHS_matrix, mass_matrix)
        """
        ny, nx = k_field.shape
        n_nodes = nx * ny
        
        # Grid spacing
        dx = grid_x[1] - grid_x[0] if len(grid_x) > 1 else 1.0
        dy = grid_y[1] - grid_y[0] if len(grid_y) > 1 else 1.0
        
        dx2 = dx * dx
        dy2 = dy * dy
        
        # Build stiffness matrix (Laplacian)
        row_stiff = []
        col_stiff = []
        data_stiff = []
        
        # Build mass matrix
        row_mass = []
        col_mass = []
        data_mass = []
        
        for j in range(ny):
            for i in range(nx):
                idx = j * nx + i
                
                # Mass term: ρ*cp
                mass_coeff = rho_field[j, i] * cp_field[j, i] / dt
                row_mass.append(idx)
                col_mass.append(idx)
                data_mass.append(mass_coeff)
                
                # Stiffness coefficient
                stiff_coeff = 2.0 * (k_field[j, i] / dx2 + k_field[j, i] / dy2)
                
                # x-direction
                if i > 0:
                    k_west = 0.5 * (k_field[j, i] + k_field[j, i-1])
                    row_stiff.append(idx)
                    col_stiff.append(j * nx + (i-1))
                    data_stiff.append(-k_west / dx2)
                else:
                    stiff_coeff -= k_field[j, i] / dx2
                
                if i < nx - 1:
                    k_east = 0.5 * (k_field[j, i] + k_field[j, i+1])
                    row_stiff.append(idx)
                    col_stiff.append(j * nx + (i+1))
                    data_stiff.append(-k_east / dx2)
                else:
                    stiff_coeff -= k_field[j, i] / dx2
                
                # y-direction
                if j > 0:
                    k_south = 0.5 * (k_field[j, i] + k_field[j-1, i])
                    row_stiff.append(idx)
                    col_stiff.append((j-1) * nx + i)
                    data_stiff.append(-k_south / dy2)
                else:
                    stiff_coeff -= k_field[j, i] / dy2
                
                if j < ny - 1:
                    k_north = 0.5 * (k_field[j, i] + k_field[j+1, i])
                    row_stiff.append(idx)
                    col_stiff.append((j+1) * nx + i)
                    data_stiff.append(-k_north / dy2)
                else:
                    stiff_coeff -= k_field[j, i] / dy2
                
                # Diagonal stiffness
                row_stiff.append(idx)
                col_stiff.append(idx)
                data_stiff.append(stiff_coeff)
        
        stiff = sparse.coo_matrix((data_stiff, (row_stiff, col_stiff)), 
                                 shape=(n_nodes, n_nodes)).tocsr()
        mass = sparse.coo_matrix((data_mass, (row_mass, col_mass)), 
                                shape=(n_nodes, n_nodes)).tocsr()
        
        # LHS = mass + dt*stiffness (for implicit backward Euler)
        lhs = mass + dt * stiff
        
        return lhs, mass
    
    def solve_steady_state(self,
                          k_field: np.ndarray,
                          grid_x: np.ndarray,
                          grid_y: np.ndarray,
                          heat_source: np.ndarray,
                          bc_conditions: Optional[List[ThermalBoundaryCondition]] = None,
                          verbose: bool = False) -> np.ndarray:
        """
        Solve steady-state heat diffusion: ∇·(k∇T) = -Q
        
        Returns:
            Temperature field
        """
        matrix, rhs = self.assemble_laplace_operator(k_field, grid_x, grid_y, bc_conditions)
        
        # Add heat source
        rhs -= heat_source.ravel()
        
        # Solve
        T = linalg.spsolve(matrix, rhs)
        
        if verbose:
            residual = np.linalg.norm(matrix @ T - rhs)
            print(f"Steady-state heat diffusion: residual = {residual:.3e}")
        
        return T.reshape(k_field.shape)
    
    def solve_transient(self,
                       T_old: np.ndarray,
                       k_field: np.ndarray,
                       cp_field: np.ndarray,
                       rho_field: np.ndarray,
                       grid_x: np.ndarray,
                       grid_y: np.ndarray,
                       heat_source: np.ndarray,
                       dt: float,
                       verbose: bool = False) -> np.ndarray:
        """
        Solve transient heat equation (backward Euler):
        ρ*cp*dT/dt - ∇·(k∇T) = Q
        
        T^{n+1} = (M + dt*K)^{-1} * (M * T^n + dt*Q)
        
        Returns:
            Temperature field at new time step
        """
        lhs, mass = self.assemble_transient_operator(
            k_field, cp_field, rho_field, grid_x, grid_y, dt
        )
        
        # RHS = M * T_old + dt * Q
        rhs = mass @ T_old.ravel() + dt * heat_source.ravel()
        
        # Solve
        T_new = linalg.spsolve(lhs, rhs)
        
        if verbose:
            residual = np.linalg.norm(lhs @ T_new - rhs)
            print(f"Transient heat equation: residual = {residual:.3e}, dt = {dt:.3e}")
        
        return T_new.reshape(k_field.shape)


class AdvectionDiffusionSolver:
    """Solve coupled advection-diffusion equation."""
    
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
    
    def assemble_advection_diffusion(self,
                                    velocity_x: np.ndarray,
                                    velocity_y: np.ndarray,
                                    k_field: np.ndarray,
                                    grid_x: np.ndarray,
                                    grid_y: np.ndarray,
                                    peclet_cutoff: float = 1e-10) -> Tuple[sparse.csr_matrix, np.ndarray]:
        """
        Assemble advection-diffusion operator using streamline upwinding Petrov-Galerkin (SUPG):
        ρ*cp*(v·∇T) - ∇·(k∇T) = Q
        
        Parameters:
            velocity_x, velocity_y: Velocity field
            k_field: Thermal conductivity field
            grid_x, grid_y: Grid coordinates
            peclet_cutoff: Cutoff for upwind term activation
            
        Returns:
            (matrix, rhs)
        """
        ny, nx = k_field.shape
        n_nodes = nx * ny
        
        dx = grid_x[1] - grid_x[0] if len(grid_x) > 1 else 1.0
        dy = grid_y[1] - grid_y[0] if len(grid_y) > 1 else 1.0
        
        dx2 = dx * dx
        dy2 = dy * dy
        
        row = []
        col = []
        data = []
        rhs = np.zeros(n_nodes)
        
        for j in range(ny):
            for i in range(nx):
                idx = j * nx + i
                
                # Velocity at this node
                vx = velocity_x[j, i] if velocity_x.shape == k_field.shape else 0.0
                vy = velocity_y[j, i] if velocity_y.shape == k_field.shape else 0.0
                
                # Peclet number (advection vs diffusion)
                v_mag = np.sqrt(vx**2 + vy**2)
                peclet = v_mag * dx / (k_field[j, i] + 1e-30)
                
                # Upwind coefficient (SUPG)
                tau = 0.0
                if abs(peclet) > peclet_cutoff:
                    tau = 0.5 * dx * (1.0 / np.tanh(peclet / 2.0) - 2.0 / peclet)
                
                # Diagonal coefficient (diffusion + advection)
                diag_coeff = 2.0 * (k_field[j, i] / dx2 + k_field[j, i] / dy2)
                
                # Diffusion (standard 5-point stencil)
                # x-direction
                if i > 0:
                    k_west = 0.5 * (k_field[j, i] + k_field[j, i-1])
                    row.append(idx)
                    col.append(j * nx + (i-1))
                    data.append(-k_west / dx2 - tau * vx / dx)
                else:
                    diag_coeff -= k_field[j, i] / dx2
                
                if i < nx - 1:
                    k_east = 0.5 * (k_field[j, i] + k_field[j, i+1])
                    row.append(idx)
                    col.append(j * nx + (i+1))
                    data.append(-k_east / dx2 + tau * vx / dx)
                else:
                    diag_coeff -= k_field[j, i] / dx2
                
                # y-direction
                if j > 0:
                    k_south = 0.5 * (k_field[j, i] + k_field[j-1, i])
                    row.append(idx)
                    col.append((j-1) * nx + i)
                    data.append(-k_south / dy2 - tau * vy / dy)
                else:
                    diag_coeff -= k_field[j, i] / dy2
                
                if j < ny - 1:
                    k_north = 0.5 * (k_field[j, i] + k_field[j+1, i])
                    row.append(idx)
                    col.append((j+1) * nx + i)
                    data.append(-k_north / dy2 + tau * vy / dy)
                else:
                    diag_coeff -= k_field[j, i] / dy2
                
                # Diagonal
                row.append(idx)
                col.append(idx)
                data.append(diag_coeff)
        
        matrix = sparse.coo_matrix((data, (row, col)), shape=(n_nodes, n_nodes)).tocsr()
        
        return matrix, rhs
    
    def solve_advection_diffusion(self,
                                 T_old: np.ndarray,
                                 velocity_x: np.ndarray,
                                 velocity_y: np.ndarray,
                                 k_field: np.ndarray,
                                 grid_x: np.ndarray,
                                 grid_y: np.ndarray,
                                 heat_source: np.ndarray,
                                 dt: float,
                                 verbose: bool = False) -> np.ndarray:
        """
        Solve coupled advection-diffusion with implicit treatment.
        """
        matrix, rhs = self.assemble_advection_diffusion(
            velocity_x, velocity_y, k_field, grid_x, grid_y
        )
        
        # Add heat source
        rhs -= heat_source.ravel()
        
        # Solve
        T_new = linalg.spsolve(matrix, rhs)
        
        if verbose:
            residual = np.linalg.norm(matrix @ T_new - rhs)
            print(f"Advection-diffusion: residual = {residual:.3e}")
        
        return T_new.reshape(k_field.shape)


class ThermalModel:
    """Complete thermal system coordinator."""
    
    def __init__(self,
                 grid_x: np.ndarray,
                 grid_y: np.ndarray,
                 initial_temperature: np.ndarray,
                 material_props: Optional[ThermalMaterialProperties] = None,
                 verbose: bool = False):
        """
        Initialize thermal model.
        
        Parameters:
            grid_x, grid_y: Grid coordinates
            initial_temperature: Initial temperature field (K)
            material_props: Thermal properties manager
            verbose: Print diagnostics
        """
        self.grid_x = grid_x
        self.grid_y = grid_y
        self.verbose = verbose
        
        # Initialize temperature field
        self.T_current = initial_temperature.copy()
        self.T_old = initial_temperature.copy()
        
        # Material properties
        if material_props is None:
            material_props = ThermalMaterialProperties()
        self.material_props = material_props
        
        # Solvers
        self.diffusion_solver = HeatDiffusionSolver(verbose=verbose)
        self.advdiff_solver = AdvectionDiffusionSolver(verbose=verbose)
        
        # History
        self.time_history = [0.0]
        self.temperature_history = [initial_temperature.copy()]
    
    def set_boundary_conditions(self,
                               bc_list: List[ThermalBoundaryCondition]):
        """Set boundary conditions."""
        self.bcs = {bc.boundary: bc for bc in bc_list}
    
    def apply_dirichlet_bc(self,
                          T: np.ndarray,
                          bc: ThermalBoundaryCondition) -> np.ndarray:
        """Apply Dirichlet boundary condition."""
        T_mod = T.copy()
        
        if bc.boundary == 'top':
            T_mod[-1, :] = bc.value
        elif bc.boundary == 'bottom':
            T_mod[0, :] = bc.value
        elif bc.boundary == 'left':
            T_mod[:, 0] = bc.value
        elif bc.boundary == 'right':
            T_mod[:, -1] = bc.value
        
        return T_mod
    
    def solve_step(self,
                   phase_field: np.ndarray,
                   velocity_x: Optional[np.ndarray] = None,
                   velocity_y: Optional[np.ndarray] = None,
                   heat_source: Optional[np.ndarray] = None,
                   dt: float = 1e5,
                   use_advection: bool = False) -> ThermalFieldData:
        """
        Solve thermal equation for one time step.
        
        Parameters:
            phase_field: Material phase field
            velocity_x, velocity_y: Velocity field (for advection)
            heat_source: Heat generation rate (W/m³)
            dt: Time step (s)
            use_advection: Include advection term
            
        Returns:
            ThermalFieldData with solution
        """
        # Get material properties on grid
        k_field = self.material_props.get_conductivity_field(phase_field)
        cp_field = self.material_props.get_capacity_field(phase_field)
        rho_field = self.material_props.get_density_field(phase_field)
        
        # Default: no heat source
        if heat_source is None:
            heat_source = np.zeros_like(self.T_current)
        
        # Solve based on mode
        if use_advection and velocity_x is not None and velocity_y is not None:
            # Advection-diffusion
            T_new = self.advdiff_solver.solve_advection_diffusion(
                self.T_old, velocity_x, velocity_y, k_field,
                self.grid_x, self.grid_y, heat_source, dt, self.verbose
            )
        else:
            # Pure diffusion (backward Euler)
            T_new = self.diffusion_solver.solve_transient(
                self.T_old, k_field, cp_field, rho_field,
                self.grid_x, self.grid_y, heat_source, dt, self.verbose
            )
        
        # Apply Dirichlet BCs if set
        if hasattr(self, 'bcs'):
            for bc in self.bcs.values():
                if bc.bc_type == ThermalBoundaryCondition.BCType.DIRICHLET:
                    T_new = self.apply_dirichlet_bc(T_new, bc)
        
        # Update history
        self.T_old = self.T_current.copy()
        self.T_current = T_new
        self.time_history.append(self.time_history[-1] + dt)
        self.temperature_history.append(T_new.copy())
        
        # Compute heat flux
        dy = self.grid_y[1] - self.grid_y[0] if len(self.grid_y) > 1 else 1.0
        dx = self.grid_x[1] - self.grid_x[0] if len(self.grid_x) > 1 else 1.0
        
        dT_dx = np.gradient(T_new, axis=1, edge_order=2) / dx
        dT_dy = np.gradient(T_new, axis=0, edge_order=2) / dy
        
        heat_flux_x = -k_field * dT_dx
        heat_flux_y = -k_field * dT_dy
        
        # Build result
        result = ThermalFieldData(
            temperature=T_new,
            temperature_old=self.T_old,
            heat_flux_x=heat_flux_x,
            heat_flux_y=heat_flux_y,
            heat_generation=heat_source,
            time=self.time_history[-1],
            dt=dt
        )
        
        return result
    
    def estimate_thermal_time_scale(self) -> float:
        """
        Estimate thermal diffusion time scale: τ ~ L²/(k/ρcp)
        
        Returns:
            Time scale (s)
        """
        L = max(self.grid_x[-1] - self.grid_x[0],
                self.grid_y[-1] - self.grid_y[0])
        
        # Use average properties
        props = self.material_props.get_properties(1)
        thermal_diffusivity = props.k / (props.rho * props.cp)
        
        tau = L * L / thermal_diffusivity
        return tau


def compute_thermal_conductivity(temperature: float,
                                mineral: str = 'olivine',
                                pressure: float = 0.0) -> float:
    """
    Compute temperature-dependent thermal conductivity.
    
    Empirical relations for common minerals:
    k(T) = k0 / (1 + β*T)  or similar
    
    Parameters:
        temperature: Temperature (K)
        mineral: 'olivine', 'basalt', 'granite', etc.
        pressure: Pressure (Pa)
        
    Returns:
        Thermal conductivity (W/m/K)
    """
    T = temperature
    
    if mineral == 'olivine':
        # Typical crustal/mantle olivine
        k0 = 5.0  # W/m/K at surface
        beta = 0.002  # 1/K
        k = k0 / (1.0 + beta * (T - 273.15))
    
    elif mineral == 'basalt':
        # Basaltic crust
        k0 = 3.5
        beta = 0.003
        k = k0 / (1.0 + beta * (T - 273.15))
    
    elif mineral == 'granite':
        # Continental crust
        k0 = 2.8
        beta = 0.002
        k = k0 / (1.0 + beta * (T - 273.15))
    
    else:
        # Default
        k = 3.0
    
    # Clamp to reasonable range
    k = np.clip(k, 0.5, 10.0)
    
    return k


def compute_heat_capacity(temperature: float,
                         mineral: str = 'olivine') -> float:
    """
    Compute temperature-dependent heat capacity.
    
    Parameters:
        temperature: Temperature (K)
        mineral: Mineral type
        
    Returns:
        Heat capacity (J/kg/K)
    """
    T = temperature
    
    if mineral == 'olivine':
        # Debye model: cp ~ T^3 at low T, cp ~ const at high T
        cp0 = 1100.0  # J/kg/K at high T
        T_debye = 600.0  # Debye temperature (K)
        cp = cp0 * (1.0 + 0.1 * (T_debye / (T + 1e-30))**2)
    
    elif mineral == 'basalt':
        cp0 = 1200.0
        T_debye = 500.0
        cp = cp0 * (1.0 + 0.08 * (T_debye / (T + 1e-30))**2)
    
    elif mineral == 'granite':
        cp0 = 1000.0
        T_debye = 400.0
        cp = cp0 * (1.0 + 0.12 * (T_debye / (T + 1e-30))**2)
    
    else:
        cp = 1000.0
    
    # Clamp
    cp = np.clip(cp, 700.0, 2000.0)
    
    return cp


def estimate_thermal_time_scale(L: float,
                               k: float = 3.0,
                               rho: float = 2800.0,
                               cp: float = 1000.0) -> float:
    """
    Estimate thermal diffusion time scale: τ = L² / α
    
    where α = k / (ρ*cp) is thermal diffusivity
    
    Parameters:
        L: Length scale (m)
        k: Thermal conductivity (W/m/K)
        rho: Density (kg/m³)
        cp: Heat capacity (J/kg/K)
        
    Returns:
        Time scale (s)
    """
    alpha = k / (rho * cp)
    tau = L * L / alpha
    return tau


def interpolate_temperature_to_markers(T_grid: np.ndarray,
                                      marker_positions: np.ndarray,
                                      grid_x: np.ndarray,
                                      grid_y: np.ndarray) -> np.ndarray:
    """
    Interpolate temperature from grid to marker positions.
    
    Uses bilinear interpolation.
    
    Parameters:
        T_grid: Temperature on grid
        marker_positions: Marker positions (n_markers, 2) with [x, y]
        grid_x, grid_y: Grid coordinates
        
    Returns:
        Temperature at marker positions
    """
    n_markers = marker_positions.shape[0]
    T_markers = np.zeros(n_markers)
    
    for m in range(n_markers):
        x_m, y_m = marker_positions[m]
        
        # Find grid cell containing marker
        i = np.searchsorted(grid_x, x_m) - 1
        j = np.searchsorted(grid_y, y_m) - 1
        
        # Clamp to valid range
        i = np.clip(i, 0, len(grid_x) - 2)
        j = np.clip(j, 0, len(grid_y) - 2)
        
        # Bilinear interpolation
        dx = (x_m - grid_x[i]) / (grid_x[i+1] - grid_x[i] + 1e-30)
        dy = (y_m - grid_y[j]) / (grid_y[j+1] - grid_y[j] + 1e-30)
        
        dx = np.clip(dx, 0.0, 1.0)
        dy = np.clip(dy, 0.0, 1.0)
        
        T00 = T_grid[j, i]
        T10 = T_grid[j, i+1]
        T01 = T_grid[j+1, i]
        T11 = T_grid[j+1, i+1]
        
        T_markers[m] = (1-dx)*(1-dy)*T00 + dx*(1-dy)*T10 + (1-dx)*dy*T01 + dx*dy*T11
    
    return T_markers
