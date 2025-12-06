"""
Tests for Linear Solver Module

Tests for:
- DirectSolver (LU factorization)
- GMRESSolver (with and without preconditioning)
- BiCGSTABSolver
- Solver selection
- Convergence and accuracy
"""

import pytest
import numpy as np
from scipy import sparse
from scipy.sparse import random, diags

from sister_py.linear_solver import (
    DirectSolver, GMRESSolver, BiCGSTABSolver,
    select_solver, SolverStats, estimate_condition_number
)


class TestDirectSolver:
    """Tests for direct sparse LU solver."""
    
    def test_direct_solver_init(self):
        """Test DirectSolver initialization."""
        solver = DirectSolver(verbose=False)
        assert solver.verbose is False
        assert solver.max_iterations == 1
        assert solver.A_lu is None
    
    def test_direct_solver_small_system(self):
        """Test solving small 5×5 SPD system."""
        # Create small SPD matrix
        A = sparse.csr_matrix(np.array([
            [4., 1., 0., 0., 0.],
            [1., 4., 1., 0., 0.],
            [0., 1., 4., 1., 0.],
            [0., 0., 1., 4., 1.],
            [0., 0., 0., 1., 4.]
        ]))
        b = np.ones(5)
        
        solver = DirectSolver(verbose=False)
        x, stats = solver.solve(A, b)
        
        assert x.shape == (5,)
        assert stats.converged
        assert stats.iterations == 1
        assert stats.solve_time > 0
        
        # Check solution accuracy
        residual = np.linalg.norm(A @ x - b)
        assert residual < 1e-10
    
    def test_direct_solver_larger_system(self):
        """Test solving larger tridiagonal system (100×100)."""
        n = 100
        A = sparse.diags(
            [np.ones(n-1), 4*np.ones(n), np.ones(n-1)],
            [-1, 0, 1],
            shape=(n, n)
        ).tocsr()
        b = np.random.rand(n)
        
        solver = DirectSolver(verbose=False)
        x, stats = solver.solve(A, b)
        
        assert x.shape == (n,)
        assert stats.converged
        
        # Verify solution
        residual_norm = np.linalg.norm(A @ x - b)
        rel_residual = residual_norm / np.linalg.norm(b)
        assert rel_residual < 1e-10
    
    def test_direct_solver_stats(self):
        """Test solver statistics."""
        A = sparse.csr_matrix(np.eye(10))
        b = np.ones(10)
        
        solver = DirectSolver()
        x, stats = solver.solve(A, b)
        
        assert isinstance(stats, SolverStats)
        assert stats.matrix_size == (10, 10)
        assert stats.matrix_nnz == 10
        assert stats.matrix_sparsity == 0.9
        assert stats.setup_time > 0
        assert stats.solve_time > 0


class TestGMRESSolver:
    """Tests for GMRES iterative solver."""
    
    def test_gmres_init(self):
        """Test GMRES initialization."""
        solver = GMRESSolver(verbose=False, max_iterations=500, restart=50)
        assert solver.max_iterations == 500
        assert solver.restart == 50
        assert solver.tolerance == 1e-6
    
    def test_gmres_no_preconditioner(self):
        """Test GMRES without preconditioning."""
        n = 50
        A = sparse.diags(
            [np.ones(n-1), 4*np.ones(n), np.ones(n-1)],
            [-1, 0, 1],
            shape=(n, n)
        ).tocsr()
        b = np.random.rand(n)
        
        solver = GMRESSolver(verbose=False, tolerance=1e-6)
        x, stats = solver.solve(A, b)
        
        assert x.shape == (n,)
        assert stats.converged
        assert stats.iterations >= 1
        
        # Verify solution
        residual_norm = np.linalg.norm(A @ x - b)
        rel_residual = residual_norm / np.linalg.norm(b)
        assert rel_residual < 1e-5
    
    def test_gmres_with_jacobi_preconditioner(self):
        """Test GMRES with Jacobi preconditioning."""
        n = 50
        # Create a more difficult problem
        A = sparse.diags(
            [np.ones(n-1), 10*np.ones(n), np.ones(n-1)],
            [-1, 0, 1],
            shape=(n, n)
        ).tocsr()
        b = np.random.rand(n)
        
        solver = GMRESSolver(verbose=False, tolerance=1e-6, preconditioner='jacobi')
        solver.setup(A)
        x, stats = solver.solve(A, b)
        
        assert stats.preconditioner_time > 0
        assert stats.converged
        
        # Solution should still be accurate
        residual_norm = np.linalg.norm(A @ x - b)
        rel_residual = residual_norm / np.linalg.norm(b)
        assert rel_residual < 1e-5
    
    def test_gmres_residual_history(self):
        """Test GMRES residual history tracking."""
        n = 30
        A = sparse.diags(
            [np.ones(n-1), 4*np.ones(n), np.ones(n-1)],
            [-1, 0, 1],
            shape=(n, n)
        ).tocsr()
        b = np.random.rand(n)
        
        solver = GMRESSolver(verbose=False, tolerance=1e-8, restart=10)
        x, stats = solver.solve(A, b)
        
        # Check residual history decreases
        assert len(stats.residual_history) > 0
        assert stats.residual_history[0] >= stats.residual_history[-1]


class TestBiCGSTABSolver:
    """Tests for BiCG-STAB iterative solver."""
    
    def test_bicgstab_init(self):
        """Test BiCG-STAB initialization."""
        solver = BiCGSTABSolver(verbose=False, max_iterations=1000)
        assert solver.max_iterations == 1000
        assert solver.tolerance == 1e-6
    
    def test_bicgstab_solve(self):
        """Test BiCG-STAB solve."""
        n = 50
        A = sparse.diags(
            [np.ones(n-1), 4*np.ones(n), np.ones(n-1)],
            [-1, 0, 1],
            shape=(n, n)
        ).tocsr()
        b = np.random.rand(n)
        
        solver = BiCGSTABSolver(verbose=False, tolerance=1e-6)
        x, stats = solver.solve(A, b)
        
        assert x.shape == (n,)
        assert stats.converged
        
        # Verify solution
        residual_norm = np.linalg.norm(A @ x - b)
        rel_residual = residual_norm / np.linalg.norm(b)
        assert rel_residual < 1e-5
    
    def test_bicgstab_with_preconditioner(self):
        """Test BiCG-STAB with preconditioning."""
        n = 50
        A = sparse.diags(
            [np.ones(n-1), 10*np.ones(n), np.ones(n-1)],
            [-1, 0, 1],
            shape=(n, n)
        ).tocsr()
        b = np.random.rand(n)
        
        solver = BiCGSTABSolver(verbose=False, tolerance=1e-6, preconditioner='jacobi')
        solver.setup(A)
        x, stats = solver.solve(A, b)
        
        assert stats.preconditioner_time > 0
        assert stats.converged


class TestSolverSelection:
    """Tests for automatic solver selection."""
    
    def test_select_solver_small_direct(self):
        """Test that small problems select direct solver."""
        solver = select_solver(661, solver_type='auto', verbose=False)
        assert isinstance(solver, DirectSolver)
    
    def test_select_solver_medium_gmres(self):
        """Test that medium problems select GMRES."""
        solver = select_solver(50000, solver_type='auto', verbose=False)
        assert isinstance(solver, GMRESSolver)
    
    def test_select_solver_explicit_direct(self):
        """Test explicit direct solver selection."""
        solver = select_solver(100000, solver_type='direct', verbose=False)
        assert isinstance(solver, DirectSolver)
    
    def test_select_solver_explicit_gmres(self):
        """Test explicit GMRES selection."""
        solver = select_solver(100, solver_type='gmres', verbose=False)
        assert isinstance(solver, GMRESSolver)
    
    def test_select_solver_explicit_bicgstab(self):
        """Test explicit BiCG-STAB selection."""
        solver = select_solver(1000, solver_type='bicgstab', verbose=False)
        assert isinstance(solver, BiCGSTABSolver)
    
    def test_select_solver_invalid_type(self):
        """Test invalid solver type raises error."""
        with pytest.raises(ValueError):
            select_solver(1000, solver_type='invalid_solver')


class TestSolverStats:
    """Tests for solver statistics."""
    
    def test_solver_stats_init(self):
        """Test SolverStats initialization."""
        stats = SolverStats()
        assert stats.converged is False
        assert stats.iterations == 0
        assert isinstance(stats.residual_history, list)
    
    def test_solver_stats_repr(self):
        """Test SolverStats string representation."""
        stats = SolverStats(converged=True, iterations=5, residual_norm=1e-6)
        repr_str = repr(stats)
        assert 'True' in repr_str
        assert '5' in repr_str
        assert 'e-06' in repr_str


class TestSaddlePointSystem:
    """Tests with saddle-point systems (like Stokes)."""
    
    def test_saddle_point_system(self):
        """Test solving a saddle-point system."""
        # Simple 2×2 block system:
        # [ K   -G^T ] [ u ]   [ f ]
        # [-G    0   ] [ p ] = [ 0 ]
        
        # K: 4×4 identity-like (velocity block)
        K = sparse.csr_matrix(np.eye(4))
        # G: 2×4 (pressure coupling)
        G = sparse.csr_matrix(np.array([[1., 0., 1., 0.],
                                        [0., 1., 0., 1.]]))
        
        # Build full system
        Z = sparse.csr_matrix((2, 2))
        A = sparse.bmat([
            [K, -G.T],
            [-G, Z]
        ]).tocsr()
        
        # RHS
        f = np.array([1., 2., 3., 4., 0., 0.])
        
        # Solve with direct solver
        solver = DirectSolver(verbose=False)
        x, stats = solver.solve(A, f)
        
        assert x.shape == (6,)
        assert stats.converged
        
        # Verify solution
        residual = np.linalg.norm(A @ x - f)
        assert residual < 1e-10
    
    def test_saddle_point_gmres(self):
        """Test GMRES on saddle-point system."""
        # Build saddle-point system (same as above)
        K = sparse.csr_matrix(np.eye(4))
        G = sparse.csr_matrix(np.array([[1., 0., 1., 0.],
                                        [0., 1., 0., 1.]]))
        Z = sparse.csr_matrix((2, 2))
        A = sparse.bmat([
            [K, -G.T],
            [-G, Z]
        ]).tocsr()
        f = np.array([1., 2., 3., 4., 0., 0.])
        
        # Solve with GMRES
        solver = GMRESSolver(verbose=False, max_iterations=100, tolerance=1e-6)
        x, stats = solver.solve(A, f)
        
        # Should converge for small system
        assert stats.converged or stats.iterations < 100
        
        residual = np.linalg.norm(A @ x - f)
        assert residual < 1e-4


class TestPerformanceComparison:
    """Tests comparing solver performance."""
    
    @pytest.mark.slow
    def test_direct_vs_gmres_small(self):
        """Compare direct vs GMRES on small problem."""
        n = 100
        A = sparse.diags(
            [np.ones(n-1), 4*np.ones(n), np.ones(n-1)],
            [-1, 0, 1],
            shape=(n, n)
        ).tocsr()
        b = np.ones(n)
        
        # Direct solver
        direct_solver = DirectSolver(verbose=False)
        x_direct, stats_direct = direct_solver.solve(A, b)
        
        # GMRES solver
        gmres_solver = GMRESSolver(verbose=False, tolerance=1e-10)
        x_gmres, stats_gmres = gmres_solver.solve(A, b)
        
        # Both should give same solution
        assert np.allclose(x_direct, x_gmres, atol=1e-8)
        
        # Direct should be faster for small systems
        assert stats_direct.solve_time < stats_gmres.solve_time * 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
