"""
SiSteR-py: Staggered-grid Iterative Solver for the Earth's Rheology in Python

A production-grade geodynamic simulation framework with fully-staggered grids,
modular rheology, and high-performance material property evaluation.
"""

from sister_py.config import ConfigurationManager, Material
from sister_py.grid import Grid, create_uniform_grid, create_zoned_grid
from sister_py.material_grid import MaterialGrid
from sister_py.solver import SolverSystem, SolverConfig, BoundaryCondition, BCType, SolutionFields
from sister_py.fd_assembly import FiniteDifferenceAssembler
from sister_py.linear_solver import (
    LinearSolver, DirectSolver, GMRESSolver, BiCGSTABSolver,
    select_solver, SolverStats
)
from sister_py.time_stepper import (
    TimeStepper, TimeIntegrator, ForwardEulerIntegrator, BackwardEulerIntegrator,
    AdaptiveTimeStep, TimeIntegrationStats, TimeStepScheme
)
from sister_py.rheology import (
    ViscosityParams, PlasticityParams, ElasticityParams, AnisotropyParams,
    ArrheniusViscosity, PlasticityYield, ElasticityModule, AnisotropicViscosity,
    RheologyModel, RheologyStress,
    compute_effective_viscosity, compute_yield_strength, estimate_max_stress
)
from sister_py.thermal_solver import (
    ThermalProperties, ThermalBoundaryCondition, ThermalMaterialProperties,
    HeatDiffusionSolver, AdvectionDiffusionSolver, ThermalModel, ThermalFieldData,
    compute_thermal_conductivity, compute_heat_capacity,
    estimate_thermal_time_scale, interpolate_temperature_to_markers
)
from sister_py.performance import (
    PerformanceProfiler, PerformanceMetrics, profile_code,
    MultiGridPreconditioner, OptimizedSolver,
    benchmark_solver, estimate_memory_usage, estimate_flops
)
from sister_py.validation import (
    PoiseuilleFlow, ThermalDiffusion, CavityFlow,
    compute_error_norms, ConvergenceStudy, ValidationReport,
    BenchmarkTestCase, run_full_validation_suite, generate_validation_report
)

__version__ = "0.2.0-phase2f"
__author__ = "SiSteR-py Contributors"

__all__ = [
    "ConfigurationManager",
    "Material",
    "Grid",
    "create_uniform_grid",
    "create_zoned_grid",
    "MaterialGrid",
    "SolverSystem",
    "SolverConfig",
    "BoundaryCondition",
    "BCType",
    "SolutionFields",
    "FiniteDifferenceAssembler",
    "LinearSolver",
    "DirectSolver",
    "GMRESSolver",
    "BiCGSTABSolver",
    "select_solver",
    "SolverStats",
    "TimeStepper",
    "TimeIntegrator",
    "ForwardEulerIntegrator",
    "BackwardEulerIntegrator",
    "AdaptiveTimeStep",
    "TimeIntegrationStats",
    "TimeStepScheme",
    "ViscosityParams",
    "PlasticityParams",
    "ElasticityParams",
    "AnisotropyParams",
    "ArrheniusViscosity",
    "PlasticityYield",
    "ElasticityModule",
    "AnisotropicViscosity",
    "RheologyModel",
    "RheologyStress",
    "compute_effective_viscosity",
    "compute_yield_strength",
    "estimate_max_stress",
    "ThermalProperties",
    "ThermalBoundaryCondition",
    "ThermalMaterialProperties",
    "HeatDiffusionSolver",
    "AdvectionDiffusionSolver",
    "ThermalModel",
    "ThermalFieldData",
    "compute_thermal_conductivity",
    "compute_heat_capacity",
    "estimate_thermal_time_scale",
    "interpolate_temperature_to_markers",
    "PerformanceProfiler",
    "PerformanceMetrics",
    "profile_code",
    "MultiGridPreconditioner",
    "OptimizedSolver",
    "benchmark_solver",
    "estimate_memory_usage",
    "estimate_flops",
    "PoiseuilleFlow",
    "ThermalDiffusion",
    "CavityFlow",
    "compute_error_norms",
    "ConvergenceStudy",
    "ValidationReport",
    "BenchmarkTestCase",
    "run_full_validation_suite",
    "generate_validation_report",
]
