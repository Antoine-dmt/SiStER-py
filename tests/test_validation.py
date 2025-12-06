"""
Tests for Validation and Benchmarking Module

Tests for:
- Analytical solutions (Poiseuille, Thermal Diffusion, Cavity Flow)
- Error computation and norms
- Convergence studies and rate estimation
- Benchmark test cases
- Validation reporting
"""

import pytest
import numpy as np

from sister_py.validation import (
    PoiseuilleFlow, ThermalDiffusion, CavityFlow,
    compute_error_norms, ConvergenceStudy, ValidationReport,
    BenchmarkTestCase, run_full_validation_suite, generate_validation_report
)


class TestPoiseuilleFlow:
    """Tests for Poiseuille flow analytical solution."""
    
    def test_poiseuille_init(self):
        """Test Poiseuille flow initialization."""
        flow = PoiseuilleFlow(U_max=2.0, h=0.5)
        assert flow.U_max == 2.0
        assert flow.h == 0.5
        assert flow.name == "Poiseuille Flow"
    
    def test_poiseuille_profile(self):
        """Test Poiseuille velocity profile."""
        flow = PoiseuilleFlow(U_max=1.0, h=1.0)
        
        # At center (y=0), should be maximum
        y_center = np.array([0.0])
        u_center = flow.evaluate(np.zeros(1), y_center)
        assert u_center[0] > 0.99
        
        # At walls (y=±1), should be zero
        y_wall = np.array([1.0])
        u_wall = flow.evaluate(np.zeros(1), y_wall)
        assert abs(u_wall[0]) < 0.01
    
    def test_poiseuille_derivatives(self):
        """Test Poiseuille derivatives."""
        flow = PoiseuilleFlow(U_max=1.0, h=1.0)
        
        # du/dx = 0 (fully developed)
        y = np.linspace(-1, 1, 10)
        du_dx = flow.evaluate_x_derivative(np.zeros_like(y), y)
        assert np.allclose(du_dx, 0.0)


class TestThermalDiffusion:
    """Tests for thermal diffusion analytical solution."""
    
    def test_thermal_init(self):
        """Test thermal diffusion initialization."""
        thermal = ThermalDiffusion(T0=300.0, T1=400.0, alpha=1e-5)
        assert thermal.T0 == 300.0
        assert thermal.T1 == 400.0
        assert thermal.alpha == 1e-5
    
    def test_thermal_initial_condition(self):
        """Test thermal diffusion at t=0."""
        thermal = ThermalDiffusion(T0=300.0, T1=400.0)
        
        x = np.linspace(0, 1, 10)
        y = np.zeros_like(x)
        
        # At t=0, should be T0 everywhere
        T = thermal.evaluate(x, y, t=0.0)
        assert np.allclose(T, thermal.T0)
    
    def test_thermal_at_boundary(self):
        """Test thermal boundary condition."""
        thermal = ThermalDiffusion(T0=300.0, T1=400.0, alpha=1e-5)
        
        x = np.array([0.0])
        y = np.zeros(1)
        t = 1.0
        
        # At x=0, should approach T1
        T_boundary = thermal.evaluate(x, y, t=t)
        assert T_boundary[0] > thermal.T0


class TestCavityFlow:
    """Tests for cavity flow analytical solution."""
    
    def test_cavity_init(self):
        """Test cavity flow initialization."""
        cavity = CavityFlow(Re=10.0)
        assert cavity.Re == 10.0
        assert cavity.name == "Lid-Driven Cavity Flow"
    
    def test_cavity_symmetry(self):
        """Test cavity flow symmetry."""
        cavity = CavityFlow(Re=100.0)
        
        # Create symmetric points
        x = np.array([0.5, 0.5])
        y = np.array([0.3, 0.7])
        
        # Stream function should be symmetric about center
        psi1 = cavity.evaluate(x[0:1], y[0:1])
        psi2 = cavity.evaluate(x[1:2], y[1:2])
        
        # Both should be positive (both in primary eddy)
        assert psi1[0] > 0
        assert psi2[0] > 0


class TestErrorMetrics:
    """Tests for error computation."""
    
    def test_compute_error_norms(self):
        """Test error norm computation."""
        numerical = np.array([1.0, 2.0, 3.0])
        analytical = np.array([1.0, 2.1, 2.9])
        
        metrics = compute_error_norms(numerical, analytical)
        
        assert metrics.L2_error >= 0
        assert metrics.Linf_error >= 0
        assert metrics.L2_error < metrics.Linf_error or np.isclose(metrics.L2_error, metrics.Linf_error)
    
    def test_error_zero_difference(self):
        """Test error when solutions are identical."""
        solution = np.array([1.0, 2.0, 3.0, 4.0])
        
        metrics = compute_error_norms(solution, solution)
        
        assert metrics.L2_error < 1e-10
        assert metrics.Linf_error < 1e-10
    
    def test_relative_error(self):
        """Test relative error computation."""
        numerical = np.array([1.1, 1.9, 3.2])
        analytical = np.array([1.0, 2.0, 3.0])
        
        metrics = compute_error_norms(numerical, analytical)
        
        assert metrics.relative_L2_error >= 0
        assert metrics.relative_L2_error < 1.0  # Less than 100%


class TestConvergenceStudy:
    """Tests for convergence studies."""
    
    def test_convergence_study_init(self):
        """Test convergence study initialization."""
        study = ConvergenceStudy("Test Problem", target_rate=2.0)
        assert study.problem_name == "Test Problem"
        assert study.target_rate == 2.0
        assert len(study.convergence_data) == 0
    
    def test_add_convergence_data(self):
        """Test adding convergence data."""
        study = ConvergenceStudy("Test")
        
        from sister_py.validation import ErrorMetrics
        error = ErrorMetrics(L2_error=1e-2, Linf_error=1e-3)
        
        study.add_data(mesh_size=10, grid_spacing=0.1, error=error)
        study.add_data(mesh_size=20, grid_spacing=0.05, error=error)
        
        assert len(study.convergence_data) == 2
        assert study.mesh_sizes == [10, 20]
    
    def test_convergence_rate_estimation(self):
        """Test convergence rate estimation."""
        study = ConvergenceStudy("Test")
        
        from sister_py.validation import ErrorMetrics
        
        # Create synthetic convergence data
        mesh_sizes = [10, 20, 40, 80]
        errors = [1e-1, 2.5e-2, 6.25e-3, 1.5625e-3]  # 2nd order
        
        for ms, err in zip(mesh_sizes, errors):
            error_metric = ErrorMetrics(L2_error=err, Linf_error=err)
            study.add_data(mesh_size=ms, grid_spacing=1.0/ms, error=error_metric)
        
        rates = study.estimate_convergence_rates()
        
        assert 'L2' in rates
        if rates['L2']:
            avg_rate = np.mean(rates['L2'])
            assert 1.5 < avg_rate < 2.5  # Should be close to 2.0


class TestValidationReport:
    """Tests for validation reports."""
    
    def test_report_init(self):
        """Test validation report initialization."""
        report = ValidationReport(test_name="Test Case")
        assert report.test_name == "Test Case"
        assert not report.accuracy_passed
        assert not report.convergence_passed
    
    def test_report_generation(self):
        """Test report generation."""
        flow = PoiseuilleFlow()
        report = ValidationReport(
            test_name="Poiseuille Flow",
            analytical_solution=flow,
            benchmark_time=0.1
        )
        
        report_str = report.generate_report()
        
        assert "Poiseuille Flow" in report_str
        assert "0.1" in report_str  # benchmark time


class TestBenchmarkTestCase:
    """Tests for benchmark test cases."""
    
    def test_poiseuille_benchmark(self):
        """Test Poiseuille flow benchmark."""
        test_case = BenchmarkTestCase("Test", "Test benchmark")
        report = test_case.run_poiseuille_benchmark(n_points=21)
        
        assert report.test_name == "Poiseuille Flow (n=21)"
        assert report.analytical_solution is not None
        assert report.error_metrics is not None
        assert report.error_metrics.L2_error < 1e-10  # Should be exact
    
    def test_thermal_convergence_benchmark(self):
        """Test thermal diffusion convergence benchmark."""
        test_case = BenchmarkTestCase("Test", "Convergence test")
        report = test_case.run_thermal_diffusion_convergence(mesh_sizes=[10, 20, 40])
        
        assert report.convergence_study is not None
        assert len(report.convergence_study.convergence_data) == 3
    
    def test_cavity_flow_benchmark(self):
        """Test cavity flow benchmark."""
        test_case = BenchmarkTestCase("Test", "Cavity test")
        report = test_case.run_cavity_flow_benchmark(Re_values=[1.0, 10.0])
        
        assert "Cavity Flow" in report.test_name
        assert report.benchmark_time >= 0


class TestFullValidationSuite:
    """Tests for full validation suite."""
    
    def test_run_full_suite(self):
        """Test running full validation suite."""
        reports = run_full_validation_suite()
        
        assert len(reports) > 0
        assert 'poiseuille' in reports
        assert 'thermal_convergence' in reports
        assert 'cavity_flow' in reports
    
    def test_generate_combined_report(self):
        """Test generating combined validation report."""
        reports = run_full_validation_suite()
        report_str = generate_validation_report(reports)
        
        assert "VALIDATION SUITE" in report_str
        assert "SUMMARY" in report_str
        assert "Passed" in report_str


class TestValidationPhysics:
    """Tests for physical consistency of validation."""
    
    def test_poiseuille_parabolic_profile(self):
        """Test Poiseuille flow has parabolic profile."""
        flow = PoiseuilleFlow(U_max=1.0, h=1.0)
        
        y = np.linspace(-1, 1, 11)
        u = flow.evaluate(np.zeros_like(y), y)
        
        # Maximum at center
        assert np.argmax(u) == 5  # Middle point
        
        # Symmetric about center
        for i in range(5):
            assert np.isclose(u[i], u[10-i])
    
    def test_thermal_diffusion_smoothing(self):
        """Test thermal diffusion smooths temperature."""
        thermal = ThermalDiffusion(T0=300.0, T1=400.0, alpha=1e-5)
        
        x = np.linspace(0, 1, 11)
        y = np.zeros_like(x)
        
        # Temperature should be monotonic from boundary
        T = thermal.evaluate(x, y, t=1.0)
        
        # Should be between T0 and T1
        assert np.all(T >= thermal.T0 - 1.0)
        assert np.all(T <= thermal.T1 + 1.0)
    
    def test_cavity_eddy_formation(self):
        """Test cavity flow forms primary eddy."""
        cavity = CavityFlow(Re=100.0)
        
        # Create fine grid in cavity
        n = 17
        x = np.linspace(0.1, 0.9, n)
        y = np.linspace(0.1, 0.9, n)
        X, Y = np.meshgrid(x, y)
        
        psi = cavity.evaluate(X, Y)
        
        # Should have non-zero stream function in interior
        assert np.max(np.abs(psi)) > 1e-6


class TestAccuracyThresholds:
    """Tests for accuracy threshold validation."""
    
    def test_excellent_accuracy(self):
        """Test excellent accuracy classification."""
        report = ValidationReport(test_name="Test")
        
        from sister_py.validation import ErrorMetrics
        report.error_metrics = ErrorMetrics(L2_error=1e-5)
        
        report_str = report.generate_report()
        assert "EXCELLENT" in report_str
        assert report.accuracy_passed
    
    def test_good_accuracy(self):
        """Test good accuracy classification."""
        report = ValidationReport(test_name="Test")
        
        from sister_py.validation import ErrorMetrics
        report.error_metrics = ErrorMetrics(L2_error=1e-3)
        
        report_str = report.generate_report()
        assert "GOOD" in report_str
        assert report.accuracy_passed
    
    def test_poor_accuracy(self):
        """Test poor accuracy classification."""
        report = ValidationReport(test_name="Test")
        
        from sister_py.validation import ErrorMetrics
        report.error_metrics = ErrorMetrics(L2_error=1.0)
        
        report_str = report.generate_report()
        assert "POOR" in report_str
        assert not report.accuracy_passed
