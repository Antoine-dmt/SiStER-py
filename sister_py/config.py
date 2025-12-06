"""
Configuration management for SiSteR-py geodynamic simulations.

This module provides:
- Pydantic-based validation of YAML configuration files
- Material property objects with viscosity and density methods
- ConfigurationManager for loading, validating, and exporting configs
"""

import os
from typing import Optional, Dict, Any
import numpy as np
import yaml
from pydantic import BaseModel, Field, field_validator


# ============================================================================
# Pydantic Configuration Models
# ============================================================================

class SimulationConfig(BaseModel):
    """Simulation time stepping configuration."""
    
    Nt: int = Field(..., gt=0, description="Total number of time steps")
    dt_out: int = Field(..., gt=0, description="Output frequency (every dt_out steps)")
    output_dir: str = Field(..., description="Output directory path")


class DomainConfig(BaseModel):
    """Computational domain dimensions."""
    
    xsize: float = Field(..., gt=0, description="Domain width in x-direction (m)")
    ysize: float = Field(..., gt=0, description="Domain height in y-direction (m)")


class GridConfig(BaseModel):
    """Grid spacing and zone boundaries configuration."""
    
    x_spacing: list[float] = Field(..., description="X-direction grid spacing per zone (m)")
    x_breaks: list[float] = Field(..., description="X-direction zone boundaries (m)")
    y_spacing: list[float] = Field(..., description="Y-direction grid spacing per zone (m)")
    y_breaks: list[float] = Field(..., description="Y-direction zone boundaries (m)")
    
    @field_validator('x_spacing', 'y_spacing')
    @classmethod
    def spacing_positive(cls, v):
        """Ensure grid spacing is positive."""
        if any(x <= 0 for x in v):
            raise ValueError('Grid spacing must be positive (> 0)')
        return v
    
    @field_validator('x_breaks', 'y_breaks')
    @classmethod
    def breaks_monotonic(cls, v):
        """Ensure zone boundaries are strictly increasing."""
        if len(v) < 2:
            raise ValueError('At least 2 zone boundaries required')
        if not all(v[i] < v[i+1] for i in range(len(v)-1)):
            raise ValueError('Zone boundaries must be strictly increasing')
        return v


class DensityParams(BaseModel):
    """Density parameters for thermal expansion model."""
    
    rho0: float = Field(..., gt=0, description="Reference density at T=0 K (kg/m³)")
    alpha: float = Field(..., description="Thermal expansion coefficient (1/K)")


class DuctileCreepParams(BaseModel):
    """Power-law creep parameters (diffusion or dislocation)."""
    
    A: float = Field(..., gt=0, description="Creep constant (Pa⁻ⁿ·s⁻¹)")
    E: float = Field(..., ge=0, description="Activation energy (J/mol)")
    n: float = Field(..., gt=0, description="Stress exponent (dimensionless)")


class RheologyConfig(BaseModel):
    """Ductile rheology configuration."""
    
    type: str = Field(..., description="Rheology type (e.g., 'ductile')")
    diffusion: Optional[DuctileCreepParams] = Field(None, description="Diffusion creep parameters")
    dislocation: Optional[DuctileCreepParams] = Field(None, description="Dislocation creep parameters")


class PlasticityParams(BaseModel):
    """Mohr-Coulomb plasticity parameters."""
    
    C: float = Field(..., ge=0, description="Cohesion (Pa)")
    mu: float = Field(..., gt=0, lt=1, description="Friction coefficient (0 < μ < 1)")


class ElasticityParams(BaseModel):
    """Elastic parameters."""
    
    G: float = Field(..., gt=0, description="Shear modulus (Pa)")


class ThermalParams(BaseModel):
    """Thermal properties."""
    
    k: float = Field(..., description="Thermal conductivity (W/m/K)")
    cp: float = Field(..., description="Specific heat capacity (J/kg/K)")


class MaterialConfig(BaseModel):
    """Complete material properties for a single phase."""
    
    phase: int = Field(..., gt=0, description="Phase ID (positive integer)")
    name: str = Field(..., description="Material name")
    density: DensityParams = Field(..., description="Density parameters")
    rheology: Optional[RheologyConfig] = Field(None, description="Ductile rheology")
    plasticity: Optional[PlasticityParams] = Field(None, description="Plasticity parameters")
    elasticity: Optional[ElasticityParams] = Field(None, description="Elasticity parameters")
    thermal: Optional[ThermalParams] = Field(None, description="Thermal parameters")


class BCConfig(BaseModel):
    """Boundary condition for a boundary segment."""
    
    type: str = Field(..., description="BC type: 'velocity' or 'stress'")
    vx: Optional[float] = Field(None, description="X-velocity (m/s)")
    vy: Optional[float] = Field(None, description="Y-velocity (m/s)")
    sxx: Optional[float] = Field(None, description="XX stress component (Pa)")
    sxy: Optional[float] = Field(None, description="XY stress component (Pa)")


class PhysicsConfig(BaseModel):
    """Physics flags for simulation."""
    
    elasticity: bool = Field(..., description="Enable elastic deformation")
    plasticity: bool = Field(..., description="Enable plastic yield")
    thermal: bool = Field(..., description="Enable thermal processes")


class SolverConfig(BaseModel):
    """Nonlinear solver configuration."""
    
    Npicard_min: int = Field(..., gt=0, description="Minimum Picard iterations")
    Npicard_max: int = Field(..., gt=0, description="Maximum Picard iterations")
    conv_tol: float = Field(..., gt=0, description="Convergence tolerance")
    switch_to_newton: int = Field(..., ge=0, description="Switch to Newton after N iterations")


class FullConfig(BaseModel):
    """Complete simulation configuration."""
    
    SIMULATION: SimulationConfig
    DOMAIN: DomainConfig
    GRID: GridConfig
    MATERIALS: list[MaterialConfig]
    BC: Dict[str, BCConfig]
    PHYSICS: PhysicsConfig
    SOLVER: SolverConfig
    
    @field_validator('MATERIALS')
    @classmethod
    def phases_unique(cls, v):
        """Ensure phase IDs are unique."""
        phases = [m.phase for m in v]
        if len(phases) != len(set(phases)):
            raise ValueError('Phase IDs must be unique')
        return v


# ============================================================================
# Material Class
# ============================================================================

class Material:
    """
    Wrapper for material properties and rheology calculations.
    
    Provides methods to compute:
    - Density as function of temperature
    - Ductile viscosity via power-law creep
    - Plastic viscosity via Mohr-Coulomb yield
    - Effective viscosity (minimum of ductile and plastic)
    """
    
    # Gas constant (J/mol/K)
    R_GAS = 8.314
    
    def __init__(self, config: MaterialConfig):
        """
        Initialize Material from config.
        
        Args:
            config: MaterialConfig instance
        """
        self.config = config
    
    @property
    def phase(self) -> int:
        """Phase ID."""
        return self.config.phase
    
    @property
    def name(self) -> str:
        """Material name."""
        return self.config.name
    
    def density(self, T: float) -> float:
        """
        Compute density with thermal expansion.
        
        Model: ρ(T) = ρ₀ * (1 - α * ΔT)
        where ΔT = T - T_ref (assume T_ref = 0 K for now)
        
        Args:
            T: Temperature (K)
        
        Returns:
            Density (kg/m³)
        """
        rho0 = self.config.density.rho0
        alpha = self.config.density.alpha
        return rho0 * (1.0 - alpha * T)
    
    def viscosity_ductile(self, sigma_II: float, eps_II: float, T: float) -> float:
        """
        Compute ductile viscosity via power-law creep.
        
        Model: ε̇ = A·σⁿ·exp(-E/RT)
        Inverted: η = 1 / (2·A·σ^(n-1)·exp(-E/RT))
        
        For combined diffusion + dislocation creep, uses harmonic mean.
        
        Args:
            sigma_II: Second invariant of deviatoric stress (Pa)
            eps_II: Second invariant of strain rate (1/s) [not used, kept for API]
            T: Temperature (K)
        
        Returns:
            Viscosity (Pa·s), or inf if no rheology defined
        
        References:
            Hirth & Kohlstedt (2003): Rheology of the upper mantle and the mantle wedge
        """
        if self.config.rheology is None:
            return float('inf')
        
        etas = []
        
        # Diffusion creep
        if self.config.rheology.diffusion and sigma_II > 0 and T > 0:
            p = self.config.rheology.diffusion
            A, E, n = p.A, p.E, p.n
            exp_term = np.exp(-E / (self.R_GAS * T))
            eta_diff = 1.0 / (2.0 * A * (sigma_II ** (n - 1.0)) * exp_term)
            etas.append(eta_diff)
        
        # Dislocation creep
        if self.config.rheology.dislocation and sigma_II > 0 and T > 0:
            p = self.config.rheology.dislocation
            A, E, n = p.A, p.E, p.n
            exp_term = np.exp(-E / (self.R_GAS * T))
            eta_disc = 1.0 / (2.0 * A * (sigma_II ** (n - 1.0)) * exp_term)
            etas.append(eta_disc)
        
        if not etas:
            return float('inf')
        elif len(etas) == 1:
            return etas[0]
        else:
            # Harmonic mean for combined creep
            return 1.0 / sum(1.0 / eta for eta in etas)
    
    def viscosity_plastic(self, sigma_II: float, P: float) -> float:
        """
        Compute plastic viscosity via Mohr-Coulomb yield criterion.
        
        Model: σ_Y = (C + μ·P)·cos(arctan(μ))
        
        Returns viscosity when stress exceeds yield, otherwise inf.
        
        Args:
            sigma_II: Second invariant of stress (Pa)
            P: Pressure (Pa, positive for compression)
        
        Returns:
            Yield viscosity (Pa·s), or inf if not yielding or no plasticity
        
        References:
            Byerlee (1978): Friction of rocks
        """
        if self.config.plasticity is None or P <= 0:
            return float('inf')
        
        C = self.config.plasticity.C
        mu = self.config.plasticity.mu
        
        sigma_Y = (C + mu * P) * np.cos(np.arctan(mu))
        
        if sigma_II > sigma_Y:
            # Simplified: return yield stress / 2 (used with strain rate in actual solver)
            return sigma_Y / 2.0
        else:
            return float('inf')
    
    def viscosity_effective(self, sigma_II: float, eps_II: float, T: float, P: float) -> float:
        """
        Compute effective viscosity.
        
        Model: η_eff = min(η_ductile, η_plastic)
        
        Args:
            sigma_II: Second invariant of deviatoric stress (Pa)
            eps_II: Second invariant of strain rate (1/s)
            T: Temperature (K)
            P: Pressure (Pa)
        
        Returns:
            Effective viscosity (Pa·s)
        """
        eta_duct = self.viscosity_ductile(sigma_II, eps_II, T)
        eta_plast = self.viscosity_plastic(sigma_II, P)
        return min(eta_duct, eta_plast)


# ============================================================================
# ConfigurationManager Class
# ============================================================================

class ConfigurationManager:
    """
    Load, validate, and manage simulation configurations.
    
    Provides:
    - YAML file loading with environment variable expansion
    - Full validation via Pydantic
    - Nested attribute access (e.g., cfg.DOMAIN.xsize)
    - Material object creation
    - Round-trip export (save/load)
    - String representation for logging
    """
    
    def __init__(self, config: FullConfig):
        """
        Initialize with validated configuration.
        
        Args:
            config: FullConfig instance
        """
        self.config = config
    
    @classmethod
    def load(cls, filepath: str) -> 'ConfigurationManager':
        """
        Load and validate configuration from YAML file.
        
        Supports environment variable expansion in paths (e.g., ${HOME}, $USER).
        
        Args:
            filepath: Path to YAML configuration file
        
        Returns:
            ConfigurationManager instance
        
        Raises:
            FileNotFoundError: If file does not exist
            yaml.YAMLError: If YAML is malformed
            ValueError: If validation fails (Pydantic)
        
        Example:
            >>> cfg = ConfigurationManager.load('config.yaml')
            >>> print(cfg.DOMAIN.xsize)
            170000.0
        """
        # Expand environment variables
        filepath = os.path.expandvars(filepath)
        
        # Load YAML
        with open(filepath, 'r') as f:
            data = yaml.safe_load(f)
        
        # Validate with Pydantic (will raise ValidationError with detailed messages)
        config = FullConfig(**data)
        
        return cls(config)
    
    def __getattr__(self, name: str) -> Any:
        """
        Provide nested attribute access.
        
        Example:
            >>> cfg.DOMAIN.xsize
            >>> cfg.MATERIALS[0].name
        """
        if hasattr(self.config, name):
            return getattr(self.config, name)
        raise AttributeError(f"ConfigurationManager has no attribute '{name}'")
    
    def get_materials(self) -> Dict[int, Material]:
        """
        Create Material objects from MATERIALS config section.
        
        Returns:
            Dictionary mapping phase_id (int) → Material instance
        
        Example:
            >>> materials = cfg.get_materials()
            >>> sticky_layer = materials[1]
            >>> mantle = materials[2]
        """
        return {m.phase: Material(m) for m in self.config.MATERIALS}
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Export configuration as nested dictionary.
        
        Returns:
            JSON-serializable dictionary
        """
        return self.config.model_dump()
    
    def to_yaml(self, filepath: str) -> None:
        """
        Save configuration to YAML file.
        
        Args:
            filepath: Output path
        
        Example:
            >>> cfg.to_yaml('output_config.yaml')
        """
        data = self.to_dict()
        with open(filepath, 'w') as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)
    
    def to_string(self) -> str:
        """
        Export configuration as formatted YAML string.
        
        Useful for logging and debugging.
        
        Returns:
            YAML-formatted string
        
        Example:
            >>> print(cfg.to_string())
        """
        return yaml.dump(self.to_dict(), default_flow_style=False, sort_keys=False)
    
    def validate(self) -> None:
        """
        Re-validate current configuration.
        
        Useful after programmatic modifications.
        
        Raises:
            ValueError: If validation fails
        """
        FullConfig(**self.to_dict())
