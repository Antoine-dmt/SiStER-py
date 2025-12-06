"""
Material Grid Module for SiSteR-py

Interpolates material properties (viscosity, density, elasticity) to grid nodes.
Supports phase-based material distributions and variable property assignment.

Classes:
    MaterialGrid: Main material property interpolation class
    MaterialProperties: Container for per-node material properties

Functions:
    interpolate_to_normal_nodes: Average properties to normal nodes
    interpolate_to_staggered_nodes: Average properties to staggered nodes
"""

import numpy as np
from typing import Dict, Tuple, List, Optional, Union
from dataclasses import dataclass, field
from pathlib import Path

from sister_py.config import ConfigurationManager, Material


@dataclass
class MaterialProperties:
    """Material properties at a single point."""
    phase_id: int
    density: float
    viscosity_ductile: float
    viscosity_plastic: float
    viscosity_effective: float
    cohesion: float
    friction_angle: float
    elastic_modulus: float


@dataclass
class MaterialGridMetadata:
    """Metadata about material grid."""
    n_materials: int
    n_nodes_x: int
    n_nodes_y: int
    property_names: List[str] = field(default_factory=list)
    min_viscosity: float = 0.0
    max_viscosity: float = 0.0
    min_density: float = 0.0
    max_density: float = 0.0
    phase_id_range: Tuple[int, int] = (0, 0)


class MaterialGrid:
    """
    Material property interpolation on fully-staggered grid.
    
    Interpolates material properties from phase markers to:
    - Normal nodes (x_n, y_n) for pressure and normal stress
    - X-staggered nodes (x_s, y_n) for x-component shear stress
    - Y-staggered nodes (x_n, y_s) for y-component shear stress
    
    Attributes:
        grid (dict): Grid coordinates from Grid.to_dict()
        cfg (ConfigurationManager): Configuration with material definitions
        materials (Dict[int, Material]): Map of phase ID to Material objects
        
        # Property arrays (n_nodes_x × n_nodes_y)
        density_n (np.ndarray): Density on normal nodes
        viscosity_ductile_n (np.ndarray): Ductile viscosity on normal nodes
        viscosity_plastic_n (np.ndarray): Plastic viscosity on normal nodes
        viscosity_effective_n (np.ndarray): Effective viscosity on normal nodes
        cohesion_n (np.ndarray): Cohesion on normal nodes
        friction_n (np.ndarray): Friction angle on normal nodes
        
        # Staggered properties
        density_xs (np.ndarray): Density on x-staggered nodes
        density_ys (np.ndarray): Density on y-staggered nodes
        viscosity_xs (np.ndarray): Viscosity on x-staggered nodes
        viscosity_ys (np.ndarray): Viscosity on y-staggered nodes
        
        # Phase distribution
        phase_n (np.ndarray): Phase ID on normal nodes
        phase_xs (np.ndarray): Phase ID on x-staggered nodes
        phase_ys (np.ndarray): Phase ID on y-staggered nodes
        
        metadata (MaterialGridMetadata): Grid statistics
    """
    
    def __init__(self, 
                 grid_dict: Dict,
                 cfg: ConfigurationManager,
                 phase_array_n: np.ndarray):
        """
        Initialize MaterialGrid with grid and material properties.
        
        Parameters:
            grid_dict: Grid dictionary from Grid.to_dict()
            cfg: ConfigurationManager with material definitions
            phase_array_n: Phase ID array on normal nodes (nx × ny)
                          Integer array with material phase IDs
        """
        self.grid_dict = grid_dict
        self.cfg = cfg
        self.x_n = grid_dict['x_n']
        self.y_n = grid_dict['y_n']
        self.x_s = grid_dict['x_s']
        self.y_s = grid_dict['y_s']
        
        # Load materials from config
        materials_dict = cfg.get_materials()
        # materials_dict is already {phase_id: Material(material_config), ...}
        self.materials = materials_dict
        
        # Store phase array
        self.phase_n = np.asarray(phase_array_n, dtype=int)
        
        # Validate shapes
        nx, ny = len(self.x_n), len(self.y_n)
        if self.phase_n.shape != (nx, ny):
            raise ValueError(
                f"phase_array_n shape {self.phase_n.shape} != grid shape ({nx}, {ny})"
            )
        
        # Initialize property arrays
        self._allocate_arrays()
        
        # Compute properties
        self._compute_properties_on_normal_nodes()
        self._interpolate_to_staggered_nodes()
        
        # Compute metadata
        self._compute_metadata()
    
    def _allocate_arrays(self) -> None:
        """Allocate property arrays."""
        nx, ny = len(self.x_n), len(self.y_n)
        nx_s, ny_s = len(self.x_s), len(self.y_s)
        
        # Normal node properties (nx × ny)
        self.density_n = np.zeros((nx, ny))
        self.viscosity_ductile_n = np.zeros((nx, ny))
        self.viscosity_plastic_n = np.zeros((nx, ny))
        self.viscosity_effective_n = np.zeros((nx, ny))
        self.cohesion_n = np.zeros((nx, ny))
        self.friction_n = np.zeros((nx, ny))
        
        # Staggered node properties
        self.density_xs = np.zeros((nx_s, ny))      # x-staggered: (nx-1 × ny)
        self.density_ys = np.zeros((nx, ny_s))      # y-staggered: (nx × ny-1)
        self.viscosity_xs = np.zeros((nx_s, ny))
        self.viscosity_ys = np.zeros((nx, ny_s))
        
        # Phase arrays
        self.phase_xs = np.zeros((nx_s, ny), dtype=int)
        self.phase_ys = np.zeros((nx, ny_s), dtype=int)
    
    def _compute_properties_on_normal_nodes(self) -> None:
        """Compute material properties on normal nodes (vectorized)."""
        # Reference conditions
        T = 273.0  # Reference temperature
        P = 1e6    # Reference pressure
        sigma_II = 1e6  # Reference stress
        eps_II = 1e-14  # Reference strain rate
        
        # Get unique phases to process
        unique_phases = np.unique(self.phase_n)
        
        for phase_id in unique_phases:
            # Get material for this phase
            if phase_id not in self.materials:
                raise ValueError(f"Phase {phase_id} not defined in materials")
            
            material = self.materials[phase_id]
            
            # Mask for this phase
            mask = self.phase_n == phase_id
            
            # Compute properties
            self.density_n[mask] = material.density(T)
            self.viscosity_ductile_n[mask] = material.viscosity_ductile(sigma_II, eps_II, T)
            self.viscosity_plastic_n[mask] = material.viscosity_plastic(sigma_II, P)
            self.viscosity_effective_n[mask] = material.viscosity_effective(
                sigma_II, eps_II, T, P
            )
            self.cohesion_n[mask] = material.config.plasticity.C if material.config.plasticity else 0
            self.friction_n[mask] = np.degrees(np.arctan(material.config.plasticity.mu)) if material.config.plasticity else 0
    
    def _interpolate_to_staggered_nodes(self) -> None:
        """Interpolate properties to staggered nodes."""
        # X-staggered nodes: average in x-direction (2-node averaging)
        # Located at (x_s[i], y_n[j]) = ((x_n[i] + x_n[i+1])/2, y_n[j])
        self.density_xs = (self.density_n[:-1, :] + self.density_n[1:, :]) / 2.0
        self.viscosity_xs = (
            self.viscosity_effective_n[:-1, :] + self.viscosity_effective_n[1:, :]
        ) / 2.0
        self.phase_xs = np.round(
            (self.phase_n[:-1, :] + self.phase_n[1:, :]) / 2.0
        ).astype(int)
        
        # Y-staggered nodes: average in y-direction (2-node averaging)
        # Located at (x_n[i], y_s[j]) = (x_n[i], (y_n[j] + y_n[j+1])/2)
        self.density_ys = (self.density_n[:, :-1] + self.density_n[:, 1:]) / 2.0
        self.viscosity_ys = (
            self.viscosity_effective_n[:, :-1] + self.viscosity_effective_n[:, 1:]
        ) / 2.0
        self.phase_ys = np.round(
            (self.phase_n[:, :-1] + self.phase_n[:, 1:]) / 2.0
        ).astype(int)
    
    def _compute_metadata(self) -> None:
        """Compute material grid metadata."""
        self.metadata = MaterialGridMetadata(
            n_materials=len(self.materials),
            n_nodes_x=len(self.x_n),
            n_nodes_y=len(self.y_n),
            property_names=['density', 'viscosity_ductile', 'viscosity_plastic', 
                           'viscosity_effective', 'cohesion', 'friction'],
            min_viscosity=float(np.min(self.viscosity_effective_n)),
            max_viscosity=float(np.max(self.viscosity_effective_n)),
            min_density=float(np.min(self.density_n)),
            max_density=float(np.max(self.density_n)),
            phase_id_range=(int(np.min(self.phase_n)), int(np.max(self.phase_n)))
        )
    
    def __repr__(self) -> str:
        """String representation."""
        m = self.metadata
        return (
            f"MaterialGrid(nx={m.n_nodes_x}, ny={m.n_nodes_y}, "
            f"n_materials={m.n_materials}, "
            f"η=[{m.min_viscosity:.2e}, {m.max_viscosity:.2e}] Pa·s)"
        )
    
    def __getitem__(self, key: str) -> np.ndarray:
        """Access property array by name."""
        properties = {
            'density_n': self.density_n,
            'density_xs': self.density_xs,
            'density_ys': self.density_ys,
            'viscosity_ductile_n': self.viscosity_ductile_n,
            'viscosity_plastic_n': self.viscosity_plastic_n,
            'viscosity_effective_n': self.viscosity_effective_n,
            'viscosity_xs': self.viscosity_xs,
            'viscosity_ys': self.viscosity_ys,
            'cohesion_n': self.cohesion_n,
            'friction_n': self.friction_n,
            'phase_n': self.phase_n,
            'phase_xs': self.phase_xs,
            'phase_ys': self.phase_ys,
        }
        
        if key not in properties:
            raise KeyError(f"Unknown property: {key}. Available: {list(properties.keys())}")
        
        return properties[key]
    
    def to_dict(self) -> Dict:
        """Export material grid as dictionary."""
        return {
            'grid_metadata': self.grid_dict['metadata'],
            'material_metadata': {
                'n_materials': self.metadata.n_materials,
                'n_nodes_x': self.metadata.n_nodes_x,
                'n_nodes_y': self.metadata.n_nodes_y,
                'properties': self.metadata.property_names,
                'viscosity_range': (self.metadata.min_viscosity, self.metadata.max_viscosity),
                'density_range': (self.metadata.min_density, self.metadata.max_density),
                'phase_range': self.metadata.phase_id_range,
            },
            'properties': {
                'density_n': self.density_n,
                'density_xs': self.density_xs,
                'density_ys': self.density_ys,
                'viscosity_ductile_n': self.viscosity_ductile_n,
                'viscosity_plastic_n': self.viscosity_plastic_n,
                'viscosity_effective_n': self.viscosity_effective_n,
                'viscosity_xs': self.viscosity_xs,
                'viscosity_ys': self.viscosity_ys,
                'cohesion_n': self.cohesion_n,
                'friction_n': self.friction_n,
            },
            'phases': {
                'phase_n': self.phase_n,
                'phase_xs': self.phase_xs,
                'phase_ys': self.phase_ys,
            }
        }
    
    @classmethod
    def generate(cls, 
                 cfg: ConfigurationManager,
                 grid_dict: Dict,
                 phase_generator=None) -> 'MaterialGrid':
        """
        Generate MaterialGrid from configuration and grid.
        
        Parameters:
            cfg: ConfigurationManager
            grid_dict: Grid dictionary from Grid.to_dict()
            phase_generator: Optional callable to generate phase distribution.
                           If None, creates uniform phase 1 grid.
                           Callable: (x, y, nx, ny) -> phase_array(nx, ny)
        
        Returns:
            MaterialGrid object
        """
        nx = grid_dict['metadata']['nx']
        ny = grid_dict['metadata']['ny']
        
        if phase_generator is None:
            # Default: uniform phase 1 throughout domain
            phase_array = np.ones((nx, ny), dtype=int)
        else:
            # Use custom generator
            phase_array = phase_generator(
                grid_dict['x_n'],
                grid_dict['y_n'],
                nx,
                ny
            )
        
        return cls(grid_dict, cfg, phase_array)


def interpolate_to_normal_nodes(
    values: np.ndarray,
    axis: int = 0,
    keep_dims: bool = False
) -> np.ndarray:
    """
    Interpolate values to normal nodes using arithmetic mean.
    
    Parameters:
        values: Array to interpolate (n1 × n2 or n × )
        axis: Axis along which to interpolate (0 or 1)
        keep_dims: Whether to keep original dimension size
    
    Returns:
        Interpolated array
    """
    if axis == 0:
        # Interpolate along x-axis: average neighboring x values
        result = (values[:-1, :] + values[1:, :]) / 2.0
    elif axis == 1:
        # Interpolate along y-axis: average neighboring y values
        result = (values[:, :-1] + values[:, 1:]) / 2.0
    else:
        raise ValueError("axis must be 0 or 1")
    
    return result


def interpolate_to_staggered_nodes(
    values_n: np.ndarray,
    axis: int
) -> np.ndarray:
    """
    Interpolate from normal nodes to staggered nodes.
    
    Parameters:
        values_n: Values on normal nodes (nx × ny or ny × nx)
        axis: Axis for staggering (0 for x-staggered, 1 for y-staggered)
    
    Returns:
        Values on staggered nodes
    """
    if axis == 0:
        # X-staggered: interpolate in x-direction
        return (values_n[:-1, :] + values_n[1:, :]) / 2.0
    elif axis == 1:
        # Y-staggered: interpolate in y-direction
        return (values_n[:, :-1] + values_n[:, 1:]) / 2.0
    else:
        raise ValueError("axis must be 0 (x-staggered) or 1 (y-staggered)")


def create_layered_phase_distribution(
    x_coords: np.ndarray,
    y_coords: np.ndarray,
    nx: int,
    ny: int,
    layer_breaks: List[float],
    layer_phases: List[int]
) -> np.ndarray:
    """
    Create phase distribution with horizontal layers.
    
    Parameters:
        x_coords: X-coordinates of normal nodes
        y_coords: Y-coordinates of normal nodes
        nx: Number of x nodes
        ny: Number of y nodes
        layer_breaks: Y-coordinate breaks for layers (must be increasing)
        layer_phases: Phase ID for each layer
    
    Returns:
        Phase array (nx × ny)
    """
    if len(layer_breaks) != len(layer_phases) + 1:
        raise ValueError(
            f"layer_breaks length {len(layer_breaks)} != layer_phases length + 1"
        )
    
    phase_array = np.zeros((nx, ny), dtype=int)
    
    for j in range(ny):
        y = y_coords[j]
        
        # Find which layer this y belongs to
        for k in range(len(layer_phases)):
            if layer_breaks[k] <= y < layer_breaks[k+1]:
                phase_array[:, j] = layer_phases[k]
                break
        else:
            # Handle edge case: last layer
            if y >= layer_breaks[-1]:
                phase_array[:, j] = layer_phases[-1]
    
    return phase_array


def create_two_phase_distribution(
    x_coords: np.ndarray,
    y_coords: np.ndarray,
    nx: int,
    ny: int,
    transition_depth: float,
    phase1: int = 1,
    phase2: int = 2
) -> np.ndarray:
    """
    Create two-phase distribution (layer over half-space).
    
    Parameters:
        x_coords: X-coordinates
        y_coords: Y-coordinates
        nx, ny: Grid dimensions
        transition_depth: Y-coordinate of phase transition
        phase1: Phase ID for layer (y > transition_depth)
        phase2: Phase ID for half-space (y < transition_depth)
    
    Returns:
        Phase array (nx × ny)
    """
    phase_array = np.zeros((nx, ny), dtype=int)
    
    for j in range(ny):
        if y_coords[j] > transition_depth:
            phase_array[:, j] = phase1
        else:
            phase_array[:, j] = phase2
    
    return phase_array
