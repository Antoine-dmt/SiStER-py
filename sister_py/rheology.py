"""
Advanced Rheology Module for SiSteR-py

Implements temperature-dependent viscosity, plasticity, elasticity, and anisotropic rheology.

Classes:
    ViscosityLaw: Abstract base for viscosity calculations
    ArrheniusViscosity: Temperature-dependent Arrhenius viscosity
    PlasticityYield: Yield criteria (Drucker-Prager, Mohr-Coulomb)
    ElasticityModule: Elastic stress accumulation and recovery
    AnisotropicViscosity: Directional viscosity dependence
    RheologyModel: Complete rheology system combining all components

Functions:
    compute_effective_viscosity: Calculate effective viscosity from all components
    compute_yield_strength: Calculate plastic yield strength
    update_elastic_stress: Update elastic stress with rate
    estimate_max_stress: Estimate maximum stress in domain
"""

import numpy as np
from typing import Tuple, Optional, Dict, Any
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import warnings


# Physical constants
BOLTZMANN = 1.380649e-23  # J/K
UNIVERSAL_GAS = 8.314472  # J/(mol·K)


@dataclass
class ViscosityParams:
    """Parameters for viscosity calculations."""
    T_ref: float = 273.15  # Reference temperature (K)
    eta_ref: float = 1e20  # Reference viscosity (Pa·s)
    E_a: float = 500e3  # Activation energy (J/mol)
    V_a: float = 0.0  # Activation volume (m³/mol) - optional
    n: float = 3.0  # Power law exponent
    A: float = 1e-16  # Prefactor for flow law (Pa^-n/s)


@dataclass
class PlasticityParams:
    """Parameters for plasticity yield criteria."""
    cohesion_0: float = 10e6  # Cohesion at surface (Pa)
    friction_angle: float = 30.0  # Friction angle (degrees)
    friction_angle_brittle: float = 30.0  # Brittle friction angle (degrees)
    friction_angle_plastic: float = 20.0  # Plastic friction angle (degrees)
    yield_strength_ref: float = 100e6  # Reference yield strength (Pa)
    pressure_dependence: bool = True  # Include pressure dependence


@dataclass
class ElasticityParams:
    """Parameters for elasticity calculations."""
    shear_modulus: float = 5e10  # Pa
    bulk_modulus: float = 1.4e11  # Pa
    relaxation_time: float = 1e10  # Relaxation time (s)
    max_stress: float = 1e8  # Maximum elastic stress (Pa)
    enable_elasticity: bool = True


@dataclass
class AnisotropyParams:
    """Parameters for anisotropic viscosity."""
    anisotropy_ratio: float = 1.0  # Ratio of max/min viscosity
    anisotropy_angle: float = 0.0  # Angle of maximum viscosity (degrees)
    enable_anisotropy: bool = False
    strain_rate_dependent: bool = False


@dataclass
class RheologyStress:
    """Complete stress state including elastic and plastic components."""
    deviatoric_stress: np.ndarray  # Deviatoric stress tensor (3x3 or components)
    elastic_stress: np.ndarray = None  # Elastic stress component
    viscous_stress: np.ndarray = None  # Viscous stress component
    pressure: float = 0.0  # Pressure (Pa)
    yield_criterion: float = 0.0  # Yield function value
    plastic_strain: float = 0.0  # Accumulated plastic strain
    
    def __post_init__(self):
        if self.elastic_stress is None:
            self.elastic_stress = np.zeros_like(self.deviatoric_stress)
        if self.viscous_stress is None:
            self.viscous_stress = np.zeros_like(self.deviatoric_stress)
    
    @property
    def stress_invariant_II(self) -> float:
        """Second invariant of deviatoric stress (for yield)."""
        # Stress invariant: (1/2) * tr(stress²)
        return 0.5 * np.sum(self.deviatoric_stress ** 2)
    
    @property
    def principal_stresses(self) -> np.ndarray:
        """Compute principal stresses from 2D stress components."""
        # For 2D: sxx, syy, sxy stored as [sxx, syy, sxy]
        if self.deviatoric_stress.size == 3:
            sxx, syy, sxy = self.deviatoric_stress
            trace = (sxx + syy) / 2.0
            diff = (sxx - syy) / 2.0
            discriminant = diff**2 + sxy**2
            s1 = trace + np.sqrt(discriminant)
            s2 = trace - np.sqrt(discriminant)
            return np.array([s1, s2])
        else:
            # For 3D or general tensors, use eigenvalues
            return np.linalg.eigvalsh(self.deviatoric_stress.reshape(3, 3))


class ViscosityLaw(ABC):
    """Abstract base class for viscosity calculations."""
    
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
    
    @abstractmethod
    def compute_viscosity(self, 
                         temperature: float,
                         strain_rate: float,
                         pressure: float = 0.0,
                         phase: int = 0) -> float:
        """
        Compute effective viscosity.
        
        Parameters:
            temperature: Temperature (K)
            strain_rate: Strain rate invariant (1/s)
            pressure: Pressure (Pa)
            phase: Material phase index
            
        Returns:
            Effective viscosity (Pa·s)
        """
        pass
    
    @abstractmethod
    def compute_viscosity_derivative(self,
                                   temperature: float,
                                   strain_rate: float,
                                   pressure: float = 0.0,
                                   phase: int = 0) -> float:
        """Compute d(viscosity)/dT for temperature derivatives."""
        pass


class ArrheniusViscosity(ViscosityLaw):
    """Temperature-dependent Arrhenius viscosity law."""
    
    def __init__(self, params: ViscosityParams, verbose: bool = False):
        super().__init__(verbose)
        self.params = params
    
    def compute_viscosity(self,
                         temperature: float,
                         strain_rate: float,
                         pressure: float = 0.0,
                         phase: int = 0) -> float:
        """
        Arrhenius viscosity: η = η_ref * exp(E_a * (1/T - 1/T_ref) / R)
        
        For dislocation creep: σ = A^(-1/n) * ε̇^(1/n) * exp(E_a/(nRT))
        """
        if strain_rate < 1e-30:
            strain_rate = 1e-30  # Avoid division by zero
        
        # Clamp temperature to reasonable range
        T_safe = np.clip(temperature, 1.0, 10000.0)
        
        # Exponential term
        exp_term = np.exp(self.params.E_a / UNIVERSAL_GAS * 
                         (1.0 / T_safe - 1.0 / self.params.T_ref))
        
        # Dislocation creep component (if A provided)
        if self.params.A > 0 and self.params.n != 1.0:
            # σ = A^(-1/n) * ε̇^(1/n)
            stress = (self.params.A ** (-1.0 / self.params.n)) * (strain_rate ** (1.0 / self.params.n))
            viscosity = stress / (strain_rate + 1e-30)
            # Apply temperature dependence
            viscosity *= exp_term
        else:
            # Simple Arrhenius viscosity
            viscosity = self.params.eta_ref * exp_term
        
        # Optional: pressure dependence (activation volume)
        if self.params.V_a > 0:
            pressure_term = np.exp(self.params.V_a * pressure / (UNIVERSAL_GAS * T_safe))
            viscosity *= pressure_term
        
        # Clamp to reasonable range - but allow wider range for different scenarios
        viscosity = np.clip(viscosity, 1e15, 1e28)
        
        return viscosity
    
    def compute_viscosity_derivative(self,
                                   temperature: float,
                                   strain_rate: float,
                                   pressure: float = 0.0,
                                   phase: int = 0) -> float:
        """d(eta)/dT - returns derivative with respect to temperature."""
        dT = 1.0  # Small perturbation
        T1 = temperature - dT/2.0
        T2 = temperature + dT/2.0
        
        eta1 = self.compute_viscosity(T1, strain_rate, pressure, phase)
        eta2 = self.compute_viscosity(T2, strain_rate, pressure, phase)
        
        return (eta2 - eta1) / dT


class PlasticityYield:
    """Plastic yield criterion (Drucker-Prager and Mohr-Coulomb)."""
    
    def __init__(self, params: PlasticityParams, verbose: bool = False):
        self.params = params
        self.verbose = verbose
    
    def drucker_prager_yield(self,
                           deviatoric_stress: np.ndarray,
                           pressure: float) -> Tuple[float, float]:
        """
        Drucker-Prager yield criterion: sqrt(J2) = C + α*P
        
        Returns:
            (yield_function, yield_strength)
        """
        # Second deviatoric stress invariant
        J2 = 0.5 * np.sum(deviatoric_stress ** 2)
        sqrt_J2 = np.sqrt(J2 + 1e-30)
        
        # Drucker-Prager parameters
        friction_angle_rad = np.radians(self.params.friction_angle)
        sin_phi = np.sin(friction_angle_rad)
        cos_phi = np.cos(friction_angle_rad)
        
        # Cohesion with depth dependence
        depth_factor = 1.0  # Could be modified with depth
        cohesion = self.params.cohesion_0 * depth_factor
        
        # DP: sqrt(J2) = C + α*P where α = sin(φ) / sqrt(3)
        # Note: positive pressure increases yield strength (confining effect)
        alpha = sin_phi / np.sqrt(3.0)
        
        yield_strength = cohesion + alpha * abs(pressure)
        yield_strength = max(yield_strength, 1e6)  # Minimum yield strength
        
        yield_function = sqrt_J2 - yield_strength
        
        return yield_function, yield_strength
    
    def mohr_coulomb_yield(self,
                          deviatoric_stress: np.ndarray,
                          pressure: float) -> Tuple[float, float]:
        """
        Mohr-Coulomb yield criterion: max|σ₁ - σ₃| = 2C + 2P*tan(φ)
        
        Returns:
            (yield_function, yield_strength)
        """
        # Principal stresses
        if deviatoric_stress.size == 3:
            sxx, syy, sxy = deviatoric_stress
            trace = (sxx + syy) / 2.0
            diff = (sxx - syy) / 2.0
            s1 = trace + np.sqrt(diff**2 + sxy**2 + 1e-30)
            s3 = trace - np.sqrt(diff**2 + sxy**2 + 1e-30)
        else:
            eigs = np.linalg.eigvalsh(deviatoric_stress.reshape(3, 3))
            s1 = eigs[-1]  # Maximum
            s3 = eigs[0]   # Minimum
        
        # Total stresses
        sigma1 = s1 + pressure
        sigma3 = s3 + pressure
        
        # Mohr-Coulomb
        friction_angle_rad = np.radians(self.params.friction_angle)
        tan_phi = np.tan(friction_angle_rad)
        
        yield_strength = self.params.cohesion_0 * tan_phi
        
        # Failure criterion
        yield_function = (sigma1 - sigma3) - 2.0 * yield_strength - 2.0 * pressure * tan_phi
        
        return yield_function, yield_strength
    
    def compute_plastic_viscosity(self,
                                 yield_function: float,
                                 strain_rate: float,
                                 reference_viscosity: float) -> float:
        """
        Compute plastic viscosity reduction based on yield function.
        
        When yield_function > 0, material yields, viscosity is reduced.
        """
        if strain_rate < 1e-30:
            strain_rate = 1e-30
        
        if yield_function <= 0:
            # Not yielding
            return reference_viscosity
        
        # Plastic viscosity (reduced)
        # Using exponential softening
        softening = np.exp(-abs(yield_function) / self.params.yield_strength_ref)
        plastic_viscosity = reference_viscosity * softening
        
        # Ensure minimum viscosity
        plastic_viscosity = max(plastic_viscosity, reference_viscosity * 0.1)
        
        return plastic_viscosity


class ElasticityModule:
    """Elastic stress accumulation and recovery (Maxwell rheology)."""
    
    def __init__(self, params: ElasticityParams, verbose: bool = False):
        self.params = params
        self.verbose = verbose
        self.stress_accumulated = {}  # Track stress per material point
    
    def update_elastic_stress(self,
                             stress_rate: np.ndarray,
                             elastic_stress: np.ndarray,
                             dt: float) -> np.ndarray:
        """
        Maxwell element: dσ_e/dt + σ_e/τ_r = 2G * dε̇/dt
        
        Parameters:
            stress_rate: Stress rate (Pa/s)
            elastic_stress: Current elastic stress
            dt: Time step
            
        Returns:
            Updated elastic stress
        """
        if not self.params.enable_elasticity:
            return elastic_stress
        
        G = self.params.shear_modulus
        tau_r = self.params.relaxation_time
        
        if tau_r <= 0:
            return elastic_stress
        
        # Explicit update: σ^{n+1} = (σ^n + 2G*dt*ε̇^n) / (1 + dt/τ_r)
        stress_new = (elastic_stress + 2.0 * G * dt * stress_rate) / (1.0 + dt / tau_r)
        
        # Clamp to maximum stress
        stress_magnitude = np.linalg.norm(stress_new)
        if stress_magnitude > self.params.max_stress:
            stress_new = stress_new * (self.params.max_stress / (stress_magnitude + 1e-30))
        
        return stress_new
    
    def compute_elastic_moduli(self,
                              pressure: float,
                              temperature: float,
                              phase: int = 0) -> Tuple[float, float]:
        """
        Compute elastic moduli (possibly temperature/pressure dependent).
        
        Returns:
            (shear_modulus, bulk_modulus)
        """
        G = self.params.shear_modulus
        K = self.params.bulk_modulus
        
        # Optional: pressure and temperature dependence
        # G_eff = G * (1 + dG_dP * P) * (1 - dG_dT * ΔT)
        
        return G, K


class AnisotropicViscosity(ViscosityLaw):
    """Anisotropic viscosity with directional dependence."""
    
    def __init__(self, base_law: ViscosityLaw, params: AnisotropyParams, 
                 verbose: bool = False):
        super().__init__(verbose)
        self.base_law = base_law
        self.params = params
    
    def compute_viscosity(self,
                         temperature: float,
                         strain_rate: float,
                         pressure: float = 0.0,
                         phase: int = 0,
                         strain_rate_tensor: Optional[np.ndarray] = None) -> float:
        """
        Compute anisotropic viscosity based on strain rate direction.
        """
        # Get isotropic base viscosity
        eta_iso = self.base_law.compute_viscosity(temperature, strain_rate, 
                                                  pressure, phase)
        
        if not self.params.enable_anisotropy or strain_rate_tensor is None:
            return eta_iso
        
        # Compute anisotropy factor based on strain rate orientation
        # This is a simplified approach - more complex implementations
        # would use fabric tensor evolution
        
        # Normalize strain rate tensor
        norm = np.linalg.norm(strain_rate_tensor)
        if norm < 1e-30:
            return eta_iso
        
        e_norm = strain_rate_tensor / norm
        
        # Angle between strain rate and anisotropy direction
        angle_rad = np.radians(self.params.anisotropy_angle)
        
        # Simple sinusoidal modulation (can be improved)
        direction = np.array([np.cos(angle_rad), np.sin(angle_rad)])
        
        # For 2D case: dot product gives orientation
        if e_norm.size >= 2:
            alignment = abs(np.dot(e_norm[:2], direction))
        else:
            alignment = 0.5
        
        # Viscosity modulation: ranges from 1.0/ratio to ratio
        ratio = self.params.anisotropy_ratio
        anisotropy_factor = 1.0 / (ratio ** alignment)  # Geometric mean approach
        
        eta_aniso = eta_iso * anisotropy_factor
        
        return eta_aniso
    
    def compute_viscosity_derivative(self,
                                    temperature: float,
                                    strain_rate: float,
                                    pressure: float = 0.0,
                                    phase: int = 0) -> float:
        """Delegate to base law."""
        return self.base_law.compute_viscosity_derivative(temperature, strain_rate,
                                                         pressure, phase)


class RheologyModel:
    """Complete rheology system combining viscosity, plasticity, and elasticity."""
    
    def __init__(self,
                 viscosity_law: ViscosityLaw,
                 plasticity: Optional[PlasticityYield] = None,
                 elasticity: Optional[ElasticityModule] = None,
                 anisotropy: Optional[AnisotropicViscosity] = None,
                 verbose: bool = False):
        self.viscosity_law = viscosity_law
        self.plasticity = plasticity
        self.elasticity = elasticity
        self.anisotropy = anisotropy
        self.verbose = verbose
        
        # History for stress and strain
        self.stress_history = {}
        self.strain_history = {}
    
    def compute_effective_viscosity(self,
                                   temperature: float,
                                   strain_rate: float,
                                   deviatoric_stress: np.ndarray,
                                   pressure: float = 0.0,
                                   phase: int = 0) -> Tuple[float, Dict[str, float]]:
        """
        Compute effective viscosity from all rheological components.
        
        Returns:
            (effective_viscosity, breakdown_dict)
        """
        breakdown = {}
        
        # Base viscosity (Arrhenius or other law)
        eta_ductile = self.viscosity_law.compute_viscosity(
            temperature, strain_rate, pressure, phase
        )
        breakdown['ductile'] = eta_ductile
        
        # Plasticity reduction
        if self.plasticity is not None:
            yield_func, _ = self.plasticity.drucker_prager_yield(
                deviatoric_stress, pressure
            )
            eta_plastic = self.plasticity.compute_plastic_viscosity(
                yield_func, strain_rate, eta_ductile
            )
            breakdown['plastic'] = eta_plastic
            eta_ductile = eta_plastic
        
        # Anisotropy modulation (if enabled)
        if self.anisotropy is not None:
            eta_ductile = self.anisotropy.compute_viscosity(
                temperature, strain_rate, pressure, phase,
                strain_rate_tensor=deviatoric_stress
            )
            breakdown['anisotropic'] = eta_ductile
        
        return eta_ductile, breakdown
    
    def compute_stress_update(self,
                             stress_old: np.ndarray,
                             strain_rate: np.ndarray,
                             dt: float,
                             temperature: float,
                             pressure: float,
                             phase: int = 0) -> RheologyStress:
        """
        Update stress accounting for all rheological components.
        
        Returns:
            Complete stress state
        """
        # Extract strain rate invariant
        strain_rate_inv = np.sqrt(0.5 * np.sum(strain_rate ** 2))
        
        # Effective viscosity
        eta_eff, breakdown = self.compute_effective_viscosity(
            temperature, strain_rate_inv, strain_rate, pressure, phase
        )
        
        # Viscous stress update: σ_new = 2η * ε̇ * dt + σ_old
        viscous_stress_new = 2.0 * eta_eff * strain_rate * dt + stress_old
        
        # Elastic stress (if enabled)
        elastic_stress_new = stress_old.copy()
        if self.elasticity is not None:
            stress_rate = strain_rate * 2.0 * eta_eff
            elastic_stress_new = self.elasticity.update_elastic_stress(
                stress_rate, stress_old, dt
            )
        
        # Yield check for plasticity
        yield_func = 0.0
        if self.plasticity is not None:
            yield_func, _ = self.plasticity.drucker_prager_yield(
                viscous_stress_new, pressure
            )
        
        # Create result
        result = RheologyStress(
            deviatoric_stress=viscous_stress_new,
            elastic_stress=elastic_stress_new,
            viscous_stress=viscous_stress_new,
            pressure=pressure,
            yield_criterion=yield_func,
            plastic_strain=0.0
        )
        
        return result
    
    def estimate_max_stress(self,
                           strain_rate_max: float,
                           viscosity_max: float,
                           temperature_min: float = 273.15) -> float:
        """
        Estimate maximum stress in domain.
        
        σ_max ≈ 2η_max * ε̇_max
        """
        max_stress = 2.0 * viscosity_max * strain_rate_max
        return max_stress


def compute_effective_viscosity(temperature: float,
                               strain_rate: float,
                               pressure: float = 0.0,
                               params: Optional[ViscosityParams] = None,
                               use_plasticity: bool = False,
                               plasticity_params: Optional[PlasticityParams] = None) -> float:
    """
    Convenience function to compute effective viscosity.
    
    Parameters:
        temperature: Temperature (K)
        strain_rate: Strain rate invariant (1/s)
        pressure: Pressure (Pa)
        params: Viscosity parameters
        use_plasticity: Include plasticity reduction
        plasticity_params: Plasticity parameters
        
    Returns:
        Effective viscosity (Pa·s)
    """
    if params is None:
        params = ViscosityParams()
    
    # Arrhenius viscosity
    arrhenius = ArrheniusViscosity(params)
    eta = arrhenius.compute_viscosity(temperature, strain_rate, pressure)
    
    # Optional plasticity reduction
    if use_plasticity and plasticity_params is not None:
        plasticity = PlasticityYield(plasticity_params)
        # Simple deviatoric stress estimate
        deviatoric_stress = np.array([eta * strain_rate, -eta * strain_rate / 2, 0.0])
        yield_func, _ = plasticity.drucker_prager_yield(deviatoric_stress, pressure)
        eta = plasticity.compute_plastic_viscosity(yield_func, strain_rate, eta)
    
    return eta


def compute_yield_strength(pressure: float,
                          params: Optional[PlasticityParams] = None,
                          criterion: str = 'drucker-prager') -> float:
    """
    Convenience function to compute yield strength.
    
    Parameters:
        pressure: Confining pressure (Pa)
        params: Plasticity parameters
        criterion: 'drucker-prager' or 'mohr-coulomb'
        
    Returns:
        Yield strength (Pa)
    """
    if params is None:
        params = PlasticityParams()
    
    plasticity = PlasticityYield(params)
    
    # Dummy deviatoric stress for yield calculation
    deviatoric_stress = np.array([1e7, 1e7, 0.0])
    
    if criterion == 'drucker-prager':
        _, yield_strength = plasticity.drucker_prager_yield(deviatoric_stress, pressure)
    elif criterion == 'mohr-coulomb':
        _, yield_strength = plasticity.mohr_coulomb_yield(deviatoric_stress, pressure)
    else:
        raise ValueError(f"Unknown yield criterion: {criterion}")
    
    return yield_strength


def estimate_max_stress(strain_rate_max: float,
                       viscosity_max: float) -> float:
    """Estimate maximum deviatoric stress: σ_max ≈ 2η * ε̇"""
    return 2.0 * viscosity_max * strain_rate_max
