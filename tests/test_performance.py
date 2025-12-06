"""
Tests for Performance Optimization Module

Tests for:
- Profiling and timing
- Multigrid preconditioner
- Optimized solvers (direct, GMRES, BiCG-STAB, multigrid)
- Benchmarking and memory estimation
"""

import pytest
import numpy as np
from scipy import sparse

from sister_py.performance import (
    PerformanceProfiler, PerformanceMetrics, profile_code,
    MultiGridPreconditioner, OptimizedSolver,
    benchmark_solver, estimate_memory_usage, estimate_flops
)


class TestPerformanceMetrics:
    """Tests for performance metrics."""
    
    def test_metrics_init(self):
        """Test metrics initialization."""
        metric = PerformanceMetrics(operation="test_op", time_elapsed=0.5)
        assert metric.operation == "test_op"
        assert metric.time_elapsed == 0.5
        assert metric.iterations == 0
    
    def test_metrics_flops(self):
        """Test FLOPS calculation."""
        metric = PerformanceMetrics(
            operation="matvec",
            nnz=1000,
            iterations=10
        )
        flops = metric.flops
        assert flops == 2 * 1000 * 10


class TestPerformanceProfiler:
    """Tests for performance profiler."""
    
    def test_profiler_timer(self):
        """Test timer context manager."""
        profiler = PerformanceProfiler()
        
        with profiler.timer("test_op"):
            np.sum(np.ones(1000))
        
        assert "test_op" in profiler.timers
        assert profiler.call_counts["test_op"] == 1
        assert profiler.timers["test_op"] > 0
    
    def test_profiler_multiple_calls(self):
        """Test multiple timer calls."""
        profiler = PerformanceProfiler()
        
        for _ in range(3):
            with profiler.timer("operation"):
                pass
        
        assert profiler.call_counts["operation"] == 3
    
    def test_profiler_summary(self):
        """Test timing summary."""
        profiler = PerformanceProfiler()
        
        with profiler.timer("op1"):
            pass
        with profiler.timer("op2"):
            pass
        
        summary = profiler.get_summary()
        assert "Performance Summary" in summary
        assert "op1" in summary
        assert "op2" in summary
    
    def test_profiler_record_metric(self):
        """Test metric recording."""
        profiler = PerformanceProfiler()
        metric = PerformanceMetrics("test", 0.1)
        
        profiler.record_metric(metric)
        assert len(profiler.metrics) == 1
        assert profiler.metrics[0] == metric
    
    def test_profiler_reset(self):
        """Test profiler reset."""
        profiler = PerformanceProfiler()
        
        with profiler.timer("op"):
            pass
        
        profiler.reset()
        assert len(profiler.timers) == 0
        assert len(profiler.call_counts) == 0


class TestProfileCodeDecorator:
    """Tests for profile_code decorator."""
    
    def test_decorator_timing(self):
        """Test decorator adds timing."""
        profiler = PerformanceProfiler()
        
        @profile_code(profiler)
        def slow_function():
            np.sum(np.ones(1000))
            return 42
        
        result = slow_function()
        
        assert result == 42
        assert "slow_function" in profiler.timers


class TestMultiGridPreconditioner:
    """Tests for multigrid preconditioner."""
    
    def test_mg_init(self):
        """Test multigrid initialization."""
        mg = MultiGridPreconditioner(n_levels=3, coarsen_factor=2)
        assert mg.n_levels == 3
        assert mg.coarsen_factor == 2
    
    def test_restriction_operator(self):
        """Test restriction operator creation."""
        mg = MultiGridPreconditioner(n_levels=2, coarsen_factor=2)
        
        # Create simple test matrix
        n = 16
        A = sparse.diags([2, -1, -1], [0, 1, -1], shape=(n, n))
        
        # Setup for 4x4 grid
        mg.setup(A, grid_shape=(4, 4))
        
        assert len(mg.restriction_ops) > 0
        R = mg.restriction_ops[0]
        assert R.shape[0] == 4  # Coarse grid
        assert R.shape[1] == 16  # Fine grid
    
    def test_mg_jacobi_smoothing(self):
        """Test Jacobi smoothing."""
        mg = MultiGridPreconditioner()
        
        # Simple 5x5 system
        A = sparse.diags([4, -1, -1], [0, 1, -1], shape=(5, 5))
        x = np.ones(5)
        b = np.ones(5)
        
        x_smooth = mg.apply_smoothing(A, x, b, n_smooth=1)
        
        assert x_smooth.shape == x.shape
        assert not np.allclose(x_smooth, x)  # Should change
    
    def test_mg_setup_and_solve(self):
        """Test multigrid setup and solve."""
        mg = MultiGridPreconditioner(n_levels=2, coarsen_factor=2)
        
        # Create simple Poisson-like system
        n = 16
        A = sparse.diags([4, -1, -1], [0, 1, -1], shape=(n, n))
        A = A.tocsr()
        
        b = np.ones(n)
        
        # Setup with 4x4 grid
        mg.setup(A, grid_shape=(4, 4))
        
        # Solve
        x, n_iter = mg.solve(A, b, tol=1e-4, maxiter=10)
        
        assert x.shape == b.shape
        assert n_iter > 0
        
        # Check solution
        residual = np.linalg.norm(b - A @ x)
        assert residual < 1e-3


class TestOptimizedSolver:
    """Tests for optimized solver."""
    
    def _create_test_matrix(self, size: int = 10):
        """Create simple test matrix and RHS."""
        # Poisson-like system
        A = sparse.diags([4, -1, -1], [0, 1, -1], shape=(size, size))
        A = A.tocsr()
        b = np.ones(size)
        return A, b
    
    def test_solver_init(self):
        """Test solver initialization."""
        solver = OptimizedSolver(verbose=False)
        assert solver.profiler is not None
    
    def test_direct_solve(self):
        """Test direct solver."""
        solver = OptimizedSolver()
        A, b = self._create_test_matrix(10)
        
        x, metric = solver.solve_direct(A, b)
        
        assert x.shape == b.shape
        assert metric.operation == "direct_solve"
        assert metric.time_elapsed > 0
        assert metric.residual >= 0
    
    def test_gmres_solve(self):
        """Test GMRES solver."""
        solver = OptimizedSolver()
        A, b = self._create_test_matrix(20)
        
        x, n_iter, metric = solver.solve_iterative_gmres(A, b, maxiter=50)
        
        assert x.shape == b.shape
        assert metric.operation == "gmres_solve"
        assert n_iter >= 0  # 0 means converged in first iteration
        assert n_iter <= 50
    
    def test_bicgstab_solve(self):
        """Test BiCG-STAB solver."""
        solver = OptimizedSolver()
        A, b = self._create_test_matrix(20)
        
        x, n_iter, metric = solver.solve_iterative_bicgstab(A, b, maxiter=50)
        
        assert x.shape == b.shape
        assert metric.operation == "bicgstab_solve"
        assert n_iter >= 0  # 0 means converged immediately
    
    def test_multigrid_solve(self):
        """Test multigrid solver."""
        solver = OptimizedSolver()
        
        # Create larger system for multigrid
        n = 64
        A = sparse.diags([4, -1, -1], [0, 1, -1], shape=(n, n))
        A = A.tocsr()
        b = np.ones(n)
        
        x, n_iter, metric = solver.solve_multigrid(A, b, grid_shape=(8, 8), maxiter=20)
        
        assert x.shape == b.shape
        assert metric.operation == "multigrid_solve"
        assert n_iter > 0
    
    def test_auto_select_solver_small(self):
        """Test auto solver selection for small problem."""
        solver = OptimizedSolver()
        A, b = self._create_test_matrix(10)
        
        x, metric = solver.auto_select_solver(A, b, method='auto')
        
        assert x.shape == b.shape
        assert metric.time_elapsed >= 0
    
    def test_auto_select_solver_method(self):
        """Test auto solver with explicit method."""
        solver = OptimizedSolver()
        A, b = self._create_test_matrix(20)
        
        for method in ['direct', 'gmres', 'bicgstab']:
            x, metric = solver.auto_select_solver(A, b, method=method)
            assert x.shape == b.shape
            assert metric.operation.endswith('_solve')
    
    def test_solver_profiling(self):
        """Test solver profiling."""
        solver = OptimizedSolver()
        A, b = self._create_test_matrix(15)
        
        x1, _ = solver.solve_direct(A, b)
        x2, _ = solver.solve_direct(A, b)
        
        summary = solver.profiler.get_summary()
        assert "direct_solve" in summary


class TestBenchmarking:
    """Tests for benchmarking functions."""
    
    def test_benchmark_solver(self):
        """Test solver benchmarking."""
        # Small system for quick testing
        n = 20
        A = sparse.diags([4, -1, -1], [0, 1, -1], shape=(n, n))
        A = A.tocsr()
        b = np.ones(n)
        
        results = benchmark_solver(A, b, methods=['direct', 'gmres'])
        
        assert 'direct' in results
        assert 'gmres' in results
        
        for method, metric in results.items():
            assert metric.time_elapsed >= 0
            assert metric.residual >= 0


class TestMemoryEstimation:
    """Tests for memory estimation."""
    
    def test_estimate_memory_small(self):
        """Test memory estimation for small problem."""
        memory_mb = estimate_memory_usage(
            matrix_shape=(100, 100),
            nnz=500,
            n_rhs=1
        )
        
        assert memory_mb > 0
        assert memory_mb < 10  # Should be small
    
    def test_estimate_memory_large(self):
        """Test memory estimation for large problem."""
        memory_mb = estimate_memory_usage(
            matrix_shape=(10000, 10000),
            nnz=100000,
            n_rhs=5,
            include_workspace=True
        )
        
        assert memory_mb > 0
        assert memory_mb < 1000  # Reasonable upper bound
    
    def test_estimate_memory_scaling(self):
        """Test memory scaling with problem size."""
        mem1 = estimate_memory_usage((100, 100), 500)
        mem2 = estimate_memory_usage((1000, 1000), 5000)
        
        # Should scale roughly as matrix size
        assert mem2 > mem1


class TestFlopEstimation:
    """Tests for FLOP estimation."""
    
    def test_flops_single_iteration(self):
        """Test FLOPS for single iteration."""
        flops = estimate_flops(matrix_size=1000, nnz=5000, n_iterations=1)
        
        # 2*nnz + 4*matrix_size + 10*matrix_size ≈ 10000 + 14000
        assert flops > 0
        assert flops < 100000
    
    def test_flops_multiple_iterations(self):
        """Test FLOPS for multiple iterations."""
        flops_1 = estimate_flops(matrix_size=1000, nnz=5000, n_iterations=1)
        flops_10 = estimate_flops(matrix_size=1000, nnz=5000, n_iterations=10)
        
        # Should scale linearly with iterations
        assert flops_10 > flops_1
        assert flops_10 >= 10 * flops_1 * 0.9  # Allow small overhead


class TestPerformancePhysics:
    """Tests for performance consistency with physics."""
    
    def test_solver_accuracy(self):
        """Test that solvers maintain accuracy."""
        # Create symmetric positive definite system
        n = 30
        A = sparse.diags([4, -1, -1], [0, 1, -1], shape=(n, n))
        A = A.tocsr()
        b = np.random.rand(n)
        
        solver = OptimizedSolver()
        
        x_direct, _ = solver.solve_direct(A, b)
        
        # All iterative solvers should give similar results
        x_gmres, _, _ = solver.solve_iterative_gmres(A, b, tol=1e-6)
        
        residual_direct = np.linalg.norm(b - A @ x_direct)
        residual_gmres = np.linalg.norm(b - A @ x_gmres)
        
        assert residual_direct < 1e-5
        assert residual_gmres < 1e-5
    
    def test_multigrid_convergence(self):
        """Test multigrid converges faster for smooth problems."""
        n = 64
        
        # Create Poisson problem
        A = sparse.diags([4, -1, -1], [0, 1, -1], shape=(n, n))
        A = A.tocsr()
        b = np.sin(np.linspace(0, np.pi, n))
        
        # Multigrid should converge in few iterations
        solver = OptimizedSolver()
        x, n_iter, _ = solver.solve_multigrid(A, b, grid_shape=(8, 8), maxiter=10)
        
        # Should converge in less than 10 iterations
        assert n_iter < 10
        
        residual = np.linalg.norm(b - A @ x)
        assert residual < 1e-4
