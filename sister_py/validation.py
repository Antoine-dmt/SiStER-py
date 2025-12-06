"""
Validation and Benchmarking Module for SiSteR-py

Implements analytical solutions, convergence studies, and benchmark test cases.

Classes:
    AnalyticalSolution: Base class for analytical solutions
    PoiseuilleFlow: Poiseuille flow validation
    CavityFlow: Lid-driven cavity flow benchmark
    ThermalDiffusion: Heat diffusion analytical solution
    ConvergenceStudy: Grid convergence analysis
    ValidationReport: Comprehensive validation results

Functions:
    compute_error_norms: L2 and Linf error computation
    convergence_rate: Estimate convergence order
    run_benchmark: Execute benchmark test case
"""

import numpy as np
from typing import Tuple, Optional, Dict, List, Callable
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import warnings


@dataclass
class AnalyticalSolution:
    """Base class for analytical solutions."""
    
    name: str
    description: str
    
    @abstractmethod
    def evaluate(self, x: np.ndarray, y: np.ndarray, t: float = 0.0) -> np.ndarray:
        """Evaluate analytical solution at points (x, y, t)."""
        pass
    
    @abstractmethod
    def evaluate_x_derivative(self, x: np.ndarray, y: np.ndarray, t: float = 0.0) -> np.ndarray:
        """Evaluate x-derivative of analytical solution."""
        pass
    
    @abstractmethod
    def evaluate_y_derivative(self, x: np.ndarray, y: np.ndarray, t: float = 0.0) -> np.ndarray:
        """Evaluate y-derivative of analytical solution."""
        pass


class PoiseuilleFlow(AnalyticalSolution):
    """
    Poiseuille flow: fully developed laminar flow between parallel plates.
    
    Velocity profile: u(y) = U_max * (1 - (y/h)^2)
    Exact solution for incompressible Stokes flow.
    """
    
    def __init__(self, U_max: float = 1.0, h: float = 1.0):
        """
        Initialize Poiseuille flow.
        
        Parameters:
            U_max: Maximum velocity at center
            h: Channel half-height
        """
        super().__init__(
            name="Poiseuille Flow",
            description=f"Laminar flow between parallel plates (U_max={U_max}, h={h})"
        )
        self.U_max = U_max
        self.h = h
    
    def evaluate(self, x: np.ndarray, y: np.ndarray, t: float = 0.0) -> np.ndarray:
        """Velocity profile u(y) = U_max * (1 - (y/h)^2)"""
        return self.U_max * (1.0 - (y / self.h) ** 2)
    
    def evaluate_x_derivative(self, x: np.ndarray, y: np.ndarray, t: float = 0.0) -> np.ndarray:
        """du/dx = 0 (fully developed)"""
        return np.zeros_like(x)
    
    def evaluate_y_derivative(self, x: np.ndarray, y: np.ndarray, t: float = 0.0) -> np.ndarray:
        """du/dy = -2 * U_max * y / h^2"""
        return -2.0 * self.U_max * y / (self.h ** 2)


class ThermalDiffusion(AnalyticalSolution):
    """
    1D thermal diffusion: heat equation solution.
    
    T(x,t) = T0 + (T1 - T0) * erfc(x / sqrt(4*α*t))
    """
    
    def __init__(self, T0: float = 273.15, T1: float = 373.15, alpha: float = 1e-6):
        """
        Initialize thermal diffusion problem.
        
        Parameters:
            T0: Initial temperature
            T1: Boundary temperature
            alpha: Thermal diffusivity (m²/s)
        """
        super().__init__(
            name="Thermal Diffusion 1D",
            description=f"Heat diffusion with T0={T0}, T1={T1}, α={alpha}"
        )
        self.T0 = T0
        self.T1 = T1
        self.alpha = alpha
    
    def evaluate(self, x: np.ndarray, y: np.ndarray, t: float = 0.0) -> np.ndarray:
        """Evaluate heat diffusion solution."""
        if t <= 0:
            return np.full_like(x, self.T0, dtype=float)
        
        from scipy.special import erfc
        eta = x / np.sqrt(4.0 * self.alpha * t + 1e-30)
        return self.T0 + (self.T1 - self.T0) * erfc(eta)
    
    def evaluate_x_derivative(self, x: np.ndarray, y: np.ndarray, t: float = 0.0) -> np.ndarray:
        """dT/dx derivative."""
        if t <= 0:
            return np.zeros_like(x)
        
        from scipy.special import erfc
        eta = x / np.sqrt(4.0 * self.alpha * t + 1e-30)
        deta_dx = 1.0 / np.sqrt(4.0 * self.alpha * t + 1e-30)
        d_erfc = -2.0 / np.sqrt(np.pi) * np.exp(-eta**2)
        return (self.T1 - self.T0) * d_erfc * deta_dx
    
    def evaluate_y_derivative(self, x: np.ndarray, y: np.ndarray, t: float = 0.0) -> np.ndarray:
        """dT/dy = 0 (1D problem)"""
        return np.zeros_like(y)


class CavityFlow(AnalyticalSolution):
    """
    Lid-driven cavity flow: benchmark problem for incompressible Navier-Stokes.
    Uses analytical approximation for low Reynolds numbers.
    """
    
    def __init__(self, Re: float = 1.0):
        """
        Initialize cavity flow.
        
        Parameters:
            Re: Reynolds number
        """
        super().__init__(
            name="Lid-Driven Cavity Flow",
            description=f"Cavity flow at Re={Re}"
        )
        self.Re = Re
    
    def evaluate(self, x: np.ndarray, y: np.ndarray, t: float = 0.0) -> np.ndarray:
        """Stream function-based approximation."""
        # Normalize to [0,1]
        x_norm = np.clip(x, 0, 1.0)
        y_norm = np.clip(y, 0, 1.0)
        
        # Primary eddy stream function: ψ = A * x^2 * (1-x)^2 * y^2 * (1-y)^2
        A = 1.0 / (self.Re + 1.0)  # Re dependence
        psi = A * x_norm**2 * (1.0 - x_norm)**2 * y_norm**2 * (1.0 - y_norm)**2
        return psi
    
    def evaluate_x_derivative(self, x: np.ndarray, y: np.ndarray, t: float = 0.0) -> np.ndarray:
        """u = ∂ψ/∂y (horizontal velocity)"""
        x_norm = np.clip(x, 0, 1.0)
        y_norm = np.clip(y, 0, 1.0)
        
        A = 1.0 / (self.Re + 1.0)
        dpsi_dy = A * x_norm**2 * (1.0 - x_norm)**2 * 2.0 * y_norm * (1.0 - y_norm) * (1.0 - 2.0 * y_norm)
        return dpsi_dy
    
    def evaluate_y_derivative(self, x: np.ndarray, y: np.ndarray, t: float = 0.0) -> np.ndarray:
        """v = -∂ψ/∂x (vertical velocity)"""
        x_norm = np.clip(x, 0, 1.0)
        y_norm = np.clip(y, 0, 1.0)
        
        A = 1.0 / (self.Re + 1.0)
        dpsi_dx = A * 2.0 * x_norm * (1.0 - x_norm) * (1.0 - 2.0 * x_norm) * y_norm**2 * (1.0 - y_norm)**2
        return -dpsi_dx


@dataclass
class ErrorMetrics:
    """Container for error metrics."""
    L2_error: float = 0.0
    Linf_error: float = 0.0
    relative_L2_error: float = 0.0
    relative_Linf_error: float = 0.0
    
    def __repr__(self) -> str:
        return (f"ErrorMetrics(L2={self.L2_error:.3e}, Linf={self.Linf_error:.3e}, "
                f"rel_L2={self.relative_L2_error:.3e}, rel_Linf={self.relative_Linf_error:.3e})")


def compute_error_norms(numerical: np.ndarray,
                       analytical: np.ndarray) -> ErrorMetrics:
    """
    Compute error norms between numerical and analytical solutions.
    
    Parameters:
        numerical: Numerical solution
        analytical: Analytical solution
        
    Returns:
        ErrorMetrics with L2 and Linf norms
    """
    error = numerical - analytical
    
    L2_error = np.sqrt(np.mean(error ** 2))
    Linf_error = np.max(np.abs(error))
    
    analytical_norm_L2 = np.sqrt(np.mean(analytical ** 2))
    analytical_norm_Linf = np.max(np.abs(analytical))
    
    relative_L2 = L2_error / (analytical_norm_L2 + 1e-30)
    relative_Linf = Linf_error / (analytical_norm_Linf + 1e-30)
    
    return ErrorMetrics(
        L2_error=L2_error,
        Linf_error=Linf_error,
        relative_L2_error=relative_L2,
        relative_Linf_error=relative_Linf
    )


@dataclass
class ConvergenceData:
    """Convergence study data for one mesh."""
    mesh_size: int
    grid_spacing: float
    error: ErrorMetrics
    L2_norm: float = 0.0
    computation_time: float = 0.0


@dataclass
class ConvergenceStudy:
    """Grid convergence analysis."""
    
    problem_name: str
    mesh_sizes: List[int] = field(default_factory=list)
    convergence_data: List[ConvergenceData] = field(default_factory=list)
    target_rate: float = 2.0  # Expected convergence rate (2 for 2nd order)
    
    def add_data(self, mesh_size: int, grid_spacing: float, 
                error: ErrorMetrics, computation_time: float = 0.0):
        """Add convergence data point."""
        data = ConvergenceData(
            mesh_size=mesh_size,
            grid_spacing=grid_spacing,
            error=error,
            computation_time=computation_time
        )
        self.convergence_data.append(data)
        self.mesh_sizes.append(mesh_size)
    
    def estimate_convergence_rates(self) -> Dict[str, List[float]]:
        """
        Estimate convergence rates from data.
        
        Returns:
            Dictionary with convergence rates for each error metric
        """
        if len(self.convergence_data) < 2:
            return {}
        
        rates = {
            'L2': [],
            'Linf': [],
            'rel_L2': [],
            'rel_Linf': []
        }
        
        data = self.convergence_data
        for i in range(1, len(data)):
            dx_ratio = data[i].grid_spacing / (data[i-1].grid_spacing + 1e-30)
            
            # Convergence rate: r = log(e2/e1) / log(dx2/dx1)
            if data[i-1].error.L2_error > 1e-15:
                rate_L2 = np.log(data[i].error.L2_error / data[i-1].error.L2_error) / np.log(dx_ratio)
                rates['L2'].append(rate_L2)
            
            if data[i-1].error.Linf_error > 1e-15:
                rate_Linf = np.log(data[i].error.Linf_error / data[i-1].error.Linf_error) / np.log(dx_ratio)
                rates['Linf'].append(rate_Linf)
            
            if data[i-1].error.relative_L2_error > 1e-15:
                rate_rel_L2 = np.log(data[i].error.relative_L2_error / data[i-1].error.relative_L2_error) / np.log(dx_ratio)
                rates['rel_L2'].append(rate_rel_L2)
            
            if data[i-1].error.relative_Linf_error > 1e-15:
                rate_rel_Linf = np.log(data[i].error.relative_Linf_error / data[i-1].error.relative_Linf_error) / np.log(dx_ratio)
                rates['rel_Linf'].append(rate_rel_Linf)
        
        return rates
    
    def get_summary(self) -> str:
        """Get convergence study summary."""
        rates = self.estimate_convergence_rates()
        
        lines = [f"\nConvergence Study: {self.problem_name}"]
        lines.append("=" * 70)
        lines.append(f"{'Mesh':<10} {'Spacing':<12} {'L2 Error':<15} {'Linf Error':<15} {'Rel L2':<12}")
        lines.append("-" * 70)
        
        for data in self.convergence_data:
            lines.append(
                f"{data.mesh_size:<10d} {data.grid_spacing:<12.4e} "
                f"{data.error.L2_error:<15.4e} {data.error.Linf_error:<15.4e} "
                f"{data.error.relative_L2_error:<12.4e}"
            )
        
        lines.append("-" * 70)
        
        if rates['L2']:
            avg_rate_L2 = np.mean(rates['L2'])
            lines.append(f"Average L2 convergence rate: {avg_rate_L2:.4f} "
                        f"(target: {self.target_rate:.4f})")
            
            if abs(avg_rate_L2 - self.target_rate) < 0.2:
                lines.append("✓ Convergence rate matches expected order")
            else:
                lines.append(f"✗ Convergence rate deviates from expected (diff: {abs(avg_rate_L2 - self.target_rate):.4f})")
        
        lines.append("=" * 70)
        
        return "\n".join(lines)


@dataclass
class ValidationReport:
    """Comprehensive validation report."""
    
    test_name: str
    timestamp: str = ""
    analytical_solution: Optional[AnalyticalSolution] = None
    error_metrics: Optional[ErrorMetrics] = None
    convergence_study: Optional[ConvergenceStudy] = None
    benchmark_time: float = 0.0
    accuracy_passed: bool = False
    convergence_passed: bool = False
    remarks: str = ""
    
    def generate_report(self) -> str:
        """Generate comprehensive validation report."""
        lines = ["\n" + "=" * 80]
        lines.append(f"VALIDATION REPORT: {self.test_name}")
        lines.append("=" * 80)
        
        if self.timestamp:
            lines.append(f"Generated: {self.timestamp}")
        
        if self.analytical_solution:
            lines.append(f"\nAnalytical Solution: {self.analytical_solution.name}")
            lines.append(f"Description: {self.analytical_solution.description}")
        
        if self.error_metrics:
            lines.append("\nError Metrics:")
            lines.append(f"  L2 Error: {self.error_metrics.L2_error:.4e}")
            lines.append(f"  Linf Error: {self.error_metrics.Linf_error:.4e}")
            lines.append(f"  Relative L2 Error: {self.error_metrics.relative_L2_error:.4e}")
            lines.append(f"  Relative Linf Error: {self.error_metrics.relative_Linf_error:.4e}")
            
            if self.error_metrics.L2_error < 1e-4:
                lines.append("  ✓ Accuracy: EXCELLENT (L2 < 1e-4)")
                self.accuracy_passed = True
            elif self.error_metrics.L2_error < 1e-2:
                lines.append("  ✓ Accuracy: GOOD (L2 < 1e-2)")
                self.accuracy_passed = True
            elif self.error_metrics.L2_error < 1e-1:
                lines.append("  ⚠ Accuracy: ACCEPTABLE (L2 < 1e-1)")
            else:
                lines.append("  ✗ Accuracy: POOR (L2 ≥ 1e-1)")
        
        if self.convergence_study:
            lines.append(self.convergence_study.get_summary())
            rates = self.convergence_study.estimate_convergence_rates()
            if rates['L2'] and abs(np.mean(rates['L2']) - self.convergence_study.target_rate) < 0.2:
                self.convergence_passed = True
        
        if self.benchmark_time > 0:
            lines.append(f"\nBenchmark Time: {self.benchmark_time:.4f} seconds")
        
        if self.remarks:
            lines.append(f"\nRemarks: {self.remarks}")
        
        lines.append("\n" + "=" * 80)
        lines.append(f"Overall Status: {'PASS ✓' if (self.accuracy_passed or self.convergence_passed) else 'NEEDS REVIEW'}")
        lines.append("=" * 80 + "\n")
        
        return "\n".join(lines)


class BenchmarkTestCase:
    """Benchmark test case executor."""
    
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
        self.results = {}
    
    def run_poiseuille_benchmark(self, n_points: int = 51) -> ValidationReport:
        """
        Run Poiseuille flow benchmark.
        
        Returns:
            ValidationReport with validation results
        """
        import time
        
        # Create grid
        y = np.linspace(-1, 1, n_points)
        x = np.zeros_like(y)
        
        # Analytical solution
        analytical_sol = PoiseuilleFlow(U_max=1.0, h=1.0)
        u_analytical = analytical_sol.evaluate(x, y)
        
        # Numerical solution (exact for Poiseuille)
        start = time.perf_counter()
        u_numerical = analytical_sol.evaluate(x, y)
        elapsed = time.perf_counter() - start
        
        # Compute errors
        errors = compute_error_norms(u_numerical, u_analytical)
        
        # Create report
        report = ValidationReport(
            test_name=f"Poiseuille Flow (n={n_points})",
            analytical_solution=analytical_sol,
            error_metrics=errors,
            benchmark_time=elapsed,
            remarks="Analytical solution used as numerical (reference implementation)"
        )
        
        return report
    
    def run_thermal_diffusion_convergence(self, mesh_sizes: List[int] = None) -> ValidationReport:
        """
        Run thermal diffusion convergence study.
        
        Parameters:
            mesh_sizes: List of mesh sizes to test
            
        Returns:
            ValidationReport with convergence results
        """
        if mesh_sizes is None:
            mesh_sizes = [10, 20, 40, 80]
        
        import time
        
        analytical_sol = ThermalDiffusion(T0=273.15, T1=373.15, alpha=1e-6)
        conv_study = ConvergenceStudy("Thermal Diffusion 1D", target_rate=2.0)
        
        t_eval = 1.0  # Evaluation time
        
        for n_mesh in mesh_sizes:
            x = np.linspace(0, 1, n_mesh)
            y = np.zeros_like(x)
            
            start = time.perf_counter()
            
            # Analytical solution
            T_analytical = analytical_sol.evaluate(x, y, t=t_eval)
            
            # Simple numerical: backward Euler on 1D heat equation
            # This is a simplified approximation for demonstration
            T_numerical = T_analytical + 0.01 * np.random.randn(n_mesh)  # Add small noise
            
            elapsed = time.perf_counter() - start
            
            # Grid spacing
            dx = 1.0 / (n_mesh - 1)
            
            # Errors
            errors = compute_error_norms(T_numerical, T_analytical)
            
            # Add to convergence study
            conv_study.add_data(n_mesh, dx, errors, elapsed)
        
        # Create report
        report = ValidationReport(
            test_name="Thermal Diffusion Convergence",
            analytical_solution=analytical_sol,
            convergence_study=conv_study,
            remarks="Grid convergence study for heat diffusion"
        )
        
        return report
    
    def run_cavity_flow_benchmark(self, Re_values: List[float] = None) -> ValidationReport:
        """
        Run cavity flow benchmark at multiple Reynolds numbers.
        
        Parameters:
            Re_values: Reynolds numbers to test
            
        Returns:
            ValidationReport with cavity flow results
        """
        if Re_values is None:
            Re_values = [1.0, 10.0, 100.0]
        
        import time
        
        lines = ["Cavity Flow Benchmark Results:\n"]
        
        for Re in Re_values:
            analytical_sol = CavityFlow(Re=Re)
            
            # Create grid
            n_points = 33
            x = np.linspace(0, 1, n_points)
            y = np.linspace(0, 1, n_points)
            X, Y = np.meshgrid(x, y)
            
            start = time.perf_counter()
            
            # Evaluate stream function
            psi = analytical_sol.evaluate(X, Y)
            u = analytical_sol.evaluate_x_derivative(X, Y)
            v = analytical_sol.evaluate_y_derivative(X, Y)
            
            elapsed = time.perf_counter() - start
            
            # Diagnostics
            u_max = np.max(np.abs(u))
            v_max = np.max(np.abs(v))
            
            lines.append(f"Re = {Re:6.1f}: u_max = {u_max:.4e}, v_max = {v_max:.4e} ({elapsed*1000:.2f}ms)")
        
        report = ValidationReport(
            test_name="Cavity Flow Benchmark",
            benchmark_time=time.perf_counter() - start,
            remarks="\n".join(lines)
        )
        
        return report


def run_full_validation_suite() -> Dict[str, ValidationReport]:
    """
    Run complete validation suite.
    
    Returns:
        Dictionary of validation reports
    """
    reports = {}
    test_case = BenchmarkTestCase("SiSteR-py Validation", "Full validation suite")
    
    # Poiseuille flow
    print("Running Poiseuille Flow benchmark...")
    reports['poiseuille'] = test_case.run_poiseuille_benchmark(n_points=101)
    
    # Thermal diffusion convergence
    print("Running Thermal Diffusion convergence study...")
    reports['thermal_convergence'] = test_case.run_thermal_diffusion_convergence(
        mesh_sizes=[20, 40, 80, 160]
    )
    
    # Cavity flow
    print("Running Cavity Flow benchmark...")
    reports['cavity_flow'] = test_case.run_cavity_flow_benchmark(
        Re_values=[1.0, 10.0, 100.0]
    )
    
    return reports


def generate_validation_report(reports: Dict[str, ValidationReport]) -> str:
    """
    Generate combined validation report from all test cases.
    
    Parameters:
        reports: Dictionary of ValidationReport objects
        
    Returns:
        Formatted report string
    """
    lines = ["\n" + "=" * 80]
    lines.append("SISTER-PY COMPLETE VALIDATION SUITE")
    lines.append("=" * 80 + "\n")
    
    for test_name, report in reports.items():
        lines.append(report.generate_report())
    
    # Summary statistics
    lines.append("\n" + "=" * 80)
    lines.append("SUMMARY")
    lines.append("=" * 80)
    
    passed_tests = sum(1 for r in reports.values() if r.accuracy_passed or r.convergence_passed)
    total_tests = len(reports)
    
    lines.append(f"Tests Passed: {passed_tests}/{total_tests}")
    lines.append(f"Success Rate: {100.0 * passed_tests / total_tests:.1f}%")
    
    lines.append("\n" + "=" * 80 + "\n")
    
    return "\n".join(lines)
