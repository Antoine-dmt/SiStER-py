"""
Grid Module for SiSteR-py

Generates fully-staggered grids for Stokes equation solving.
Supports zone-based variable spacing per Duretz et al. (2013).

Classes:
    Grid: Main grid generation class

Functions:
    generate_uniform_grid: Create uniform spacing grid
    generate_zoned_grid: Create grid with zone-based spacing
"""

import numpy as np
from typing import Tuple, Dict, List, Optional
from dataclasses import dataclass


@dataclass
class GridMetadata:
    """Metadata about generated grid."""
    nx: int  # Number of x-normal nodes
    ny: int  # Number of y-normal nodes
    n_cells_x: int  # Number of x-direction cells
    n_cells_y: int  # Number of y-direction cells
    x_min: float  # Domain minimum x
    x_max: float  # Domain maximum x
    y_min: float  # Domain minimum y
    y_max: float  # Domain maximum y
    dx_min: float  # Minimum cell width
    dx_max: float  # Maximum cell width
    dy_min: float  # Minimum cell height
    dy_max: float  # Maximum cell height
    aspect_ratio_max: float  # Maximum cell aspect ratio


class Grid:
    """
    Fully-staggered grid generator for geodynamic simulations.
    
    Generates coordinate arrays for:
    - Normal nodes (x_n, y_n): velocity, pressure
    - Staggered nodes (x_s, y_s): shear stress, staggered components
    
    Supports zone-based discretization with variable spacing.
    
    Attributes:
        x_n (np.ndarray): X coordinates of normal nodes (1D)
        y_n (np.ndarray): Y coordinates of normal nodes (1D)
        x_s (np.ndarray): X coordinates of staggered nodes (1D)
        y_s (np.ndarray): Y coordinates of staggered nodes (1D)
        metadata (GridMetadata): Grid statistics and properties
    """
    
    def __init__(self, 
                 x_n: np.ndarray,
                 y_n: np.ndarray,
                 x_s: np.ndarray,
                 y_s: np.ndarray,
                 domain_bounds: Tuple[float, float, float, float]):
        """
        Initialize Grid with coordinate arrays.
        
        Parameters:
            x_n: X coordinates of normal nodes
            y_n: Y coordinates of normal nodes
            x_s: X coordinates of staggered nodes
            y_s: Y coordinates of staggered nodes
            domain_bounds: Tuple (xmin, xmax, ymin, ymax)
        """
        self.x_n = np.asarray(x_n)
        self.y_n = np.asarray(y_n)
        self.x_s = np.asarray(x_s)
        self.y_s = np.asarray(y_s)
        
        self._validate_coordinates()
        self._compute_metadata(domain_bounds)
    
    def _validate_coordinates(self) -> None:
        """Validate coordinate arrays."""
        # Check monotonicity
        if not np.all(np.diff(self.x_n) > 0):
            raise ValueError("x_n coordinates not strictly increasing")
        if not np.all(np.diff(self.y_n) > 0):
            raise ValueError("y_n coordinates not strictly increasing")
        if not np.all(np.diff(self.x_s) > 0):
            raise ValueError("x_s coordinates not strictly increasing")
        if not np.all(np.diff(self.y_s) > 0):
            raise ValueError("y_s coordinates not strictly increasing")
        
        # Check staggered offset (approximately Δx/2, Δy/2)
        if len(self.x_s) != len(self.x_n) - 1:
            raise ValueError(f"x_s length {len(self.x_s)} != x_n length - 1 {len(self.x_n)-1}")
        if len(self.y_s) != len(self.y_n) - 1:
            raise ValueError(f"y_s length {len(self.y_s)} != y_n length - 1 {len(self.y_n)-1}")
    
    def _compute_metadata(self, domain_bounds: Tuple[float, float, float, float]) -> None:
        """Compute grid metadata."""
        x_min, x_max, y_min, y_max = domain_bounds
        
        dx = np.diff(self.x_n)
        dy = np.diff(self.y_n)
        
        # Compute aspect ratio per cell (not whole grid)
        # Use min spacing for conservative estimate
        dx_min = float(np.min(dx))
        dy_min = float(np.min(dy))
        aspect_ratio_max = max(dx_min / dy_min, dy_min / dx_min)
        
        self.metadata = GridMetadata(
            nx=len(self.x_n),
            ny=len(self.y_n),
            n_cells_x=len(dx),
            n_cells_y=len(dy),
            x_min=x_min,
            x_max=x_max,
            y_min=y_min,
            y_max=y_max,
            dx_min=dx_min,
            dx_max=float(np.max(dx)),
            dy_min=dy_min,
            dy_max=float(np.max(dy)),
            aspect_ratio_max=aspect_ratio_max
        )
    
    def __repr__(self) -> str:
        """String representation of grid."""
        m = self.metadata
        return (
            f"Grid(nx={m.nx}, ny={m.ny}, "
            f"dx=[{m.dx_min:.2e}, {m.dx_max:.2e}], "
            f"dy=[{m.dy_min:.2e}, {m.dy_max:.2e}], "
            f"aspect_ratio={m.aspect_ratio_max:.2f})"
        )
    
    def __getitem__(self, key: str) -> np.ndarray:
        """Get coordinate array by name."""
        if key == 'x_n':
            return self.x_n
        elif key == 'y_n':
            return self.y_n
        elif key == 'x_s':
            return self.x_s
        elif key == 'y_s':
            return self.y_s
        else:
            raise KeyError(f"Unknown coordinate: {key}")
    
    def to_dict(self) -> Dict:
        """Export grid as dictionary."""
        return {
            'x_n': self.x_n,
            'y_n': self.y_n,
            'x_s': self.x_s,
            'y_s': self.y_s,
            'metadata': {
                'nx': self.metadata.nx,
                'ny': self.metadata.ny,
                'n_cells_x': self.metadata.n_cells_x,
                'n_cells_y': self.metadata.n_cells_y,
                'bounds': (self.metadata.x_min, self.metadata.x_max, 
                          self.metadata.y_min, self.metadata.y_max),
                'dx_range': (self.metadata.dx_min, self.metadata.dx_max),
                'dy_range': (self.metadata.dy_min, self.metadata.dy_max),
                'aspect_ratio_max': self.metadata.aspect_ratio_max,
            }
        }
    
    @classmethod
    def generate(cls, cfg) -> 'Grid':
        """
        Generate grid from ConfigurationManager.
        
        Parameters:
            cfg: ConfigurationManager object with DOMAIN and GRID sections
            
        Returns:
            Grid object
        """
        # Extract domain parameters
        x_min = cfg.DOMAIN.x_min if hasattr(cfg.DOMAIN, 'x_min') else 0.0
        x_max = cfg.DOMAIN.xsize
        y_min = cfg.DOMAIN.y_min if hasattr(cfg.DOMAIN, 'y_min') else 0.0
        y_max = cfg.DOMAIN.ysize
        
        # Check for zone-based vs uniform grid
        if hasattr(cfg.GRID, 'x_breaks') and cfg.GRID.x_breaks:
            x_n = _generate_zoned_coordinates(
                x_min, x_max,
                cfg.GRID.x_breaks,
                cfg.GRID.x_spacing
            )
        else:
            x_n = np.linspace(x_min, x_max, cfg.GRID.nx)
        
        if hasattr(cfg.GRID, 'y_breaks') and cfg.GRID.y_breaks:
            y_n = _generate_zoned_coordinates(
                y_min, y_max,
                cfg.GRID.y_breaks,
                cfg.GRID.y_spacing
            )
        else:
            y_n = np.linspace(y_min, y_max, cfg.GRID.ny)
        
        # Generate staggered nodes (midpoints + offset)
        x_s = (x_n[:-1] + x_n[1:]) / 2.0
        y_s = (y_n[:-1] + y_n[1:]) / 2.0
        
        return cls(x_n, y_n, x_s, y_s, (x_min, x_max, y_min, y_max))


def _generate_zoned_coordinates(
    domain_min: float,
    domain_max: float,
    zone_breaks: List[float],
    zone_spacing: List[float]
) -> np.ndarray:
    """
    Generate coordinates with zone-based variable spacing.
    
    Parameters:
        domain_min: Minimum domain coordinate
        domain_max: Maximum domain coordinate
        zone_breaks: List of zone boundary coordinates (must be strictly increasing)
        zone_spacing: List of target spacing per zone
        
    Returns:
        1D array of coordinates
        
    Raises:
        ValueError: If breaks not strictly increasing or don't cover domain
    """
    # Validate inputs
    zone_breaks = np.asarray(zone_breaks)
    zone_spacing = np.asarray(zone_spacing)
    
    if len(zone_breaks) != len(zone_spacing) + 1:
        raise ValueError(
            f"zone_breaks length {len(zone_breaks)} != zone_spacing length + 1 {len(zone_spacing)+1}"
        )
    
    if not np.all(np.diff(zone_breaks) > 0):
        raise ValueError("zone_breaks not strictly increasing")
    
    if zone_breaks[0] < domain_min or zone_breaks[-1] > domain_max:
        raise ValueError(f"Zone breaks {zone_breaks} don't cover domain [{domain_min}, {domain_max}]")
    
    # Generate coordinates per zone
    coords = []
    
    for i in range(len(zone_breaks) - 1):
        z_start = zone_breaks[i]
        z_end = zone_breaks[i + 1]
        spacing = zone_spacing[i]
        
        # Number of cells in this zone
        n_cells = max(1, int(np.round((z_end - z_start) / spacing)))
        
        # Generate zone coordinates (don't include endpoint to avoid duplicates)
        z_coords = np.linspace(z_start, z_end, n_cells + 1)[:-1]
        coords.append(z_coords)
    
    # Add final point
    coords.append(np.array([zone_breaks[-1]]))
    
    return np.concatenate(coords)


def create_uniform_grid(
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    nx: int,
    ny: int
) -> Grid:
    """
    Create uniform spacing grid.
    
    Parameters:
        x_min, x_max: Domain bounds in x
        y_min, y_max: Domain bounds in y
        nx: Number of x nodes
        ny: Number of y nodes
        
    Returns:
        Grid object with uniform spacing
    """
    x_n = np.linspace(x_min, x_max, nx)
    y_n = np.linspace(y_min, y_max, ny)
    x_s = (x_n[:-1] + x_n[1:]) / 2.0
    y_s = (y_n[:-1] + y_n[1:]) / 2.0
    
    return Grid(x_n, y_n, x_s, y_s, (x_min, x_max, y_min, y_max))


def create_zoned_grid(
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    x_breaks: List[float],
    x_spacing: List[float],
    y_breaks: List[float],
    y_spacing: List[float]
) -> Grid:
    """
    Create grid with zone-based variable spacing.
    
    Parameters:
        x_min, x_max: Domain bounds in x
        y_min, y_max: Domain bounds in y
        x_breaks: Zone boundaries in x
        x_spacing: Target spacing per x zone
        y_breaks: Zone boundaries in y
        y_spacing: Target spacing per y zone
        
    Returns:
        Grid object with variable spacing
    """
    x_n = _generate_zoned_coordinates(x_min, x_max, x_breaks, x_spacing)
    y_n = _generate_zoned_coordinates(y_min, y_max, y_breaks, y_spacing)
    x_s = (x_n[:-1] + x_n[1:]) / 2.0
    y_s = (y_n[:-1] + y_n[1:]) / 2.0
    
    return Grid(x_n, y_n, x_s, y_s, (x_min, x_max, y_min, y_max))
