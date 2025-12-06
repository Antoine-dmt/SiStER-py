"""
Performance Optimization Module for SiSteR-py

Implements performance profiling, preconditioning, and optimization strategies.

Classes:
    PerformanceProfiler: Code profiling and timing utilities
    MultiGridPreconditioner: Multigrid preconditioner for sparse systems
    OptimizedSolver: High-performance solver with multiple backends
    PerformanceMetrics: Timing and performance data collection

Functions:
    profile_code: Decorator for performance profiling
    benchmark_solver: Compare solver performance across methods
    estimate_memory_usage: Predict memory consumption
"""

import numpy as np
from typing import Tuple, Optional, Dict, List, Callable, Any
from dataclasses import dataclass, field
from scipy import sparse
from scipy.sparse import linalg
import time
import warnings
from contextlib import contextmanager


@dataclass
class PerformanceMetrics:
    """Container for performance timing metrics."""
    operation: str
    time_elapsed: float = 0.0
    matrix_size: int = 0
    nnz: int = 0  # Number of non-zeros
    iterations: int = 0
    residual: float = 0.0
    memory_mb: float = 0.0
    throughput_gflops: float = 0.0
    
    def __repr__(self) -> str:
        return (f"PerformanceMetrics({self.operation}, "
                f"time={self.time_elapsed:.4f}s, "
                f"matrix_size={self.matrix_size}, "
                f"throughput={self.throughput_gflops:.2f} GFLOPS)")
    
    @property
    def flops(self) -> float:
        """Estimate FLOPs: 2*nnz per iteration for sparse matvec."""
        return 2 * self.nnz * max(1, self.iterations)


class PerformanceProfiler:
    """Profiling and timing utilities."""
    
    def __init__(self):
        self.timers = {}
        self.call_counts = {}
        self.metrics = []
    
    @contextmanager
    def timer(self, name: str):
        """Context manager for timing code blocks."""
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - start
            if name not in self.timers:
                self.timers[name] = 0.0
                self.call_counts[name] = 0
            self.timers[name] += elapsed
            self.call_counts[name] += 1
    
    def record_metric(self, metric: PerformanceMetrics):
        """Record a performance metric."""
        self.metrics.append(metric)
    
    def get_summary(self) -> str:
        """Get timing summary."""
        if not self.timers:
            return "No timing data collected"
        
        lines = ["Performance Summary:\n"]
        total_time = sum(self.timers.values())
        
        for name in sorted(self.timers.keys(), 
                          key=lambda x: self.timers[x], 
                          reverse=True):
            time_val = self.timers[name]
            calls = self.call_counts[name]
            pct = 100.0 * time_val / total_time if total_time > 0 else 0.0
            avg_time = time_val / calls if calls > 0 else 0.0
            
            lines.append(f"  {name:30s}: {time_val:10.4f}s "
                        f"({pct:5.1f}%) [{calls:6d} calls, "
                        f"avg={avg_time*1000:.3f}ms]")
        
        lines.append(f"\nTotal time: {total_time:.4f}s")
        return "\n".join(lines)
    
    def reset(self):
        """Reset all timers."""
        self.timers.clear()
        self.call_counts.clear()
        self.metrics.clear()


def profile_code(profiler: Optional[PerformanceProfiler] = None):
    """Decorator for profiling function execution."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            prof = profiler or PerformanceProfiler()
            with prof.timer(func.__name__):
                return func(*args, **kwargs)
        return wrapper
    return decorator


class MultiGridPreconditioner:
    """Multigrid preconditioner for sparse linear systems."""
    
    def __init__(self, n_levels: int = 3, coarsen_factor: int = 2):
        """
        Initialize multigrid preconditioner.
        
        Parameters:
            n_levels: Number of multigrid levels
            coarsen_factor: Factor for grid coarsening (2 or 4)
        """
        self.n_levels = n_levels
        self.coarsen_factor = coarsen_factor
        self.restriction_ops = []
        self.prolongation_ops = []
        self.coarse_matrices = []
    
    def setup(self, matrix: sparse.csr_matrix, grid_shape: Tuple[int, int]):
        """
        Setup multigrid hierarchy from fine grid matrix.
        
        Parameters:
            matrix: Fine grid matrix
            grid_shape: (ny, nx) grid dimensions
        """
        ny, nx = grid_shape
        current_matrix = matrix
        current_shape = grid_shape
        
        for level in range(self.n_levels - 1):
            # Create restriction operator (fine to coarse)
            restriction = self._create_restriction_operator(current_shape)
            self.restriction_ops.append(restriction)
            
            # Create prolongation operator (coarse to fine)
            prolongation = restriction.T.tocsr()
            self.prolongation_ops.append(prolongation)
            
            # Coarse grid matrix: R * A * P
            coarse_matrix = (restriction @ current_matrix @ prolongation).tocsr()
            self.coarse_matrices.append(coarse_matrix)
            
            # Update for next level
            current_matrix = coarse_matrix
            current_shape = (current_shape[0] // self.coarsen_factor,
                           current_shape[1] // self.coarsen_factor)
    
    def _create_restriction_operator(self, 
                                    grid_shape: Tuple[int, int]) -> sparse.csr_matrix:
        """
        Create restriction operator for one grid level.
        
        Uses full-weighting restriction: average coarse node from 4 fine nodes.
        """
        ny, nx = grid_shape
        ny_coarse = ny // self.coarsen_factor
        nx_coarse = nx // self.coarsen_factor
        
        n_fine = ny * nx
        n_coarse = ny_coarse * nx_coarse
        
        row = []
        col = []
        data = []
        
        # Full-weighting: average 4 fine nodes to 1 coarse node
        for jc in range(ny_coarse):
            for ic in range(nx_coarse):
                coarse_idx = jc * nx_coarse + ic
                fine_i = ic * self.coarsen_factor
                fine_j = jc * self.coarsen_factor
                
                # 2x2 stencil of fine nodes
                for dj in range(self.coarsen_factor):
                    for di in range(self.coarsen_factor):
                        fi = fine_i + di
                        fj = fine_j + dj
                        if fi < nx and fj < ny:
                            fine_idx = fj * nx + fi
                            row.append(coarse_idx)
                            col.append(fine_idx)
                            data.append(0.25)
        
        restriction = sparse.coo_matrix((data, (row, col)), 
                                       shape=(n_coarse, n_fine)).tocsr()
        return restriction
    
    def apply_smoothing(self,
                       matrix: sparse.csr_matrix,
                       x: np.ndarray,
                       b: np.ndarray,
                       n_smooth: int = 2) -> np.ndarray:
        """
        Apply Jacobi smoothing.
        
        x_{k+1} = x_k + omega * D^{-1} * (b - A*x_k)
        
        where D is diagonal of A.
        """
        omega = 0.66667  # Damping factor
        D_inv = 1.0 / (matrix.diagonal() + 1e-20)
        
        x_smooth = x.copy()
        for _ in range(n_smooth):
            residual = b - matrix @ x_smooth
            x_smooth = x_smooth + omega * D_inv * residual
        
        return x_smooth
    
    def apply_vcycle(self,
                    matrix: sparse.csr_matrix,
                    x: np.ndarray,
                    b: np.ndarray,
                    level: int = 0) -> np.ndarray:
        """Apply V-cycle iteration."""
        if level == self.n_levels - 1:
            # Coarsest level: direct solve
            x = linalg.spsolve(matrix, b)
            if x.ndim == 1:
                return x
            return x.ravel()
        
        # Pre-smoothing
        x = self.apply_smoothing(matrix, x, b, n_smooth=2)
        
        # Restrict residual to coarse grid
        residual = b - matrix @ x
        restriction = self.restriction_ops[level]
        b_coarse = restriction @ residual
        
        # Recursive coarse grid solve
        x_coarse = np.zeros(b_coarse.shape[0])
        coarse_matrix = self.coarse_matrices[level]
        x_coarse = self.apply_vcycle(coarse_matrix, x_coarse, b_coarse, level + 1)
        
        # Prolongate correction to fine grid
        prolongation = self.prolongation_ops[level]
        x = x + prolongation @ x_coarse
        
        # Post-smoothing
        x = self.apply_smoothing(matrix, x, b, n_smooth=2)
        
        return x
    
    def solve(self,
             matrix: sparse.csr_matrix,
             b: np.ndarray,
             x0: Optional[np.ndarray] = None,
             tol: float = 1e-6,
             maxiter: int = 100) -> Tuple[np.ndarray, int]:
        """
        Solve Ax=b using multigrid V-cycles.
        
        Returns:
            (solution, n_iterations)
        """
        if x0 is None:
            x = np.zeros(b.shape[0])
        else:
            x = x0.copy()
        
        for iteration in range(maxiter):
            x = self.apply_vcycle(matrix, x, b, level=0)
            
            residual = b - matrix @ x
            error = np.linalg.norm(residual)
            
            if error < tol:
                return x, iteration + 1
        
        return x, maxiter


class OptimizedSolver:
    """High-performance solver with multiple backends and strategies."""
    
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.profiler = PerformanceProfiler()
        self.last_metric = None
    
    def solve_direct(self,
                    matrix: sparse.csr_matrix,
                    b: np.ndarray) -> Tuple[np.ndarray, PerformanceMetrics]:
        """
        Direct solver using sparse LU decomposition.
        
        Returns:
            (solution, metrics)
        """
        with self.profiler.timer("direct_solve"):
            start = time.perf_counter()
            
            x = linalg.spsolve(matrix, b)
            if x.ndim > 1:
                x = x.ravel()
            
            elapsed = time.perf_counter() - start
        
        residual = np.linalg.norm(b - matrix @ x)
        metric = PerformanceMetrics(
            operation="direct_solve",
            time_elapsed=elapsed,
            matrix_size=matrix.shape[0],
            nnz=matrix.nnz,
            residual=residual
        )
        metric.throughput_gflops = (metric.flops / 1e9) / elapsed if elapsed > 0 else 0.0
        
        self.last_metric = metric
        self.profiler.record_metric(metric)
        
        return x, metric
    
    def solve_iterative_gmres(self,
                             matrix: sparse.csr_matrix,
                             b: np.ndarray,
                             x0: Optional[np.ndarray] = None,
                             tol: float = 1e-6,
                             restart: int = 30,
                             maxiter: int = None) -> Tuple[np.ndarray, int, PerformanceMetrics]:
        """
        Iterative solver: GMRES with optional preconditioning.
        
        Returns:
            (solution, n_iterations, metrics)
        """
        if maxiter is None:
            maxiter = min(matrix.shape[0], 100)
        
        with self.profiler.timer("gmres_solve"):
            start = time.perf_counter()
            
            # Jacobi preconditioner (diagonal scaling)
            diag = matrix.diagonal()
            diag_inv = 1.0 / (np.abs(diag) + 1e-20)
            M = sparse.diags(diag_inv)
            
            def callback(rk):
                pass  # For tracking iterations if needed
            
            x, info = linalg.gmres(matrix, b, x0=x0, rtol=tol, restart=restart,
                                  maxiter=maxiter, M=M, callback=callback)
            
            elapsed = time.perf_counter() - start
        
        residual = np.linalg.norm(b - matrix @ x)
        n_iter = info if info >= 0 else maxiter
        
        metric = PerformanceMetrics(
            operation="gmres_solve",
            time_elapsed=elapsed,
            matrix_size=matrix.shape[0],
            nnz=matrix.nnz,
            iterations=n_iter,
            residual=residual
        )
        metric.throughput_gflops = (metric.flops / 1e9) / elapsed if elapsed > 0 else 0.0
        
        self.last_metric = metric
        self.profiler.record_metric(metric)
        
        return x, n_iter, metric
    
    def solve_iterative_bicgstab(self,
                                matrix: sparse.csr_matrix,
                                b: np.ndarray,
                                x0: Optional[np.ndarray] = None,
                                tol: float = 1e-6,
                                maxiter: int = None) -> Tuple[np.ndarray, int, PerformanceMetrics]:
        """
        Iterative solver: BiCG-STAB with Jacobi preconditioning.
        
        Returns:
            (solution, n_iterations, metrics)
        """
        if maxiter is None:
            maxiter = min(matrix.shape[0], 100)
        
        with self.profiler.timer("bicgstab_solve"):
            start = time.perf_counter()
            
            # Jacobi preconditioner
            diag = matrix.diagonal()
            diag_inv = 1.0 / (np.abs(diag) + 1e-20)
            M = sparse.diags(diag_inv)
            
            x, info = linalg.bicgstab(matrix, b, x0=x0, rtol=tol,
                                     maxiter=maxiter, M=M)
            
            elapsed = time.perf_counter() - start
        
        residual = np.linalg.norm(b - matrix @ x)
        n_iter = info if info >= 0 else maxiter
        
        metric = PerformanceMetrics(
            operation="bicgstab_solve",
            time_elapsed=elapsed,
            matrix_size=matrix.shape[0],
            nnz=matrix.nnz,
            iterations=n_iter,
            residual=residual
        )
        metric.throughput_gflops = (metric.flops / 1e9) / elapsed if elapsed > 0 else 0.0
        
        self.last_metric = metric
        self.profiler.record_metric(metric)
        
        return x, n_iter, metric
    
    def solve_multigrid(self,
                       matrix: sparse.csr_matrix,
                       b: np.ndarray,
                       grid_shape: Tuple[int, int],
                       tol: float = 1e-6,
                       maxiter: int = 20) -> Tuple[np.ndarray, int, PerformanceMetrics]:
        """
        Multigrid solver with V-cycles.
        
        Returns:
            (solution, n_iterations, metrics)
        """
        with self.profiler.timer("multigrid_solve"):
            start = time.perf_counter()
            
            # Setup multigrid
            mg = MultiGridPreconditioner(n_levels=3)
            mg.setup(matrix, grid_shape)
            
            # Solve using V-cycles
            x0 = np.zeros(b.shape[0])
            x, n_iter = mg.solve(matrix, b, x0=x0, tol=tol, maxiter=maxiter)
            
            elapsed = time.perf_counter() - start
        
        residual = np.linalg.norm(b - matrix @ x)
        
        metric = PerformanceMetrics(
            operation="multigrid_solve",
            time_elapsed=elapsed,
            matrix_size=matrix.shape[0],
            nnz=matrix.nnz,
            iterations=n_iter,
            residual=residual
        )
        metric.throughput_gflops = (metric.flops / 1e9) / elapsed if elapsed > 0 else 0.0
        
        self.last_metric = metric
        self.profiler.record_metric(metric)
        
        return x, n_iter, metric
    
    def auto_select_solver(self,
                          matrix: sparse.csr_matrix,
                          b: np.ndarray,
                          grid_shape: Optional[Tuple[int, int]] = None,
                          method: str = 'auto') -> Tuple[np.ndarray, PerformanceMetrics]:
        """
        Automatically select best solver for this problem.
        
        Parameters:
            matrix: Sparse matrix
            b: Right-hand side
            grid_shape: Grid dimensions (for multigrid)
            method: 'auto', 'direct', 'gmres', 'bicgstab', 'multigrid'
            
        Returns:
            (solution, metrics)
        """
        n = matrix.shape[0]
        nnz = matrix.nnz
        density = nnz / (n * n)
        condition_estimate = np.linalg.norm(matrix.data) if len(matrix.data) > 0 else 1.0
        
        if self.verbose:
            print(f"Matrix: {n}x{n}, nnz={nnz}, density={density:.4f}")
        
        if method == 'auto':
            # Auto-select based on problem size and structure
            if n < 5000 or density > 0.1:
                method = 'direct'
            elif grid_shape is not None and n > 10000:
                method = 'multigrid'
            else:
                method = 'bicgstab'
        
        if method == 'direct':
            x, metric = self.solve_direct(matrix, b)
        elif method == 'gmres':
            x, _, metric = self.solve_iterative_gmres(matrix, b)
        elif method == 'bicgstab':
            x, _, metric = self.solve_iterative_bicgstab(matrix, b)
        elif method == 'multigrid':
            if grid_shape is None:
                # Estimate grid shape as square
                side = int(np.sqrt(n))
                grid_shape = (side, side)
            x, _, metric = self.solve_multigrid(matrix, b, grid_shape)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        if self.verbose:
            print(f"Selected method: {method}")
            print(f"  Time: {metric.time_elapsed:.4f}s")
            print(f"  Residual: {metric.residual:.3e}")
            print(f"  Throughput: {metric.throughput_gflops:.2f} GFLOPS")
        
        return x, metric


def benchmark_solver(matrix: sparse.csr_matrix,
                    b: np.ndarray,
                    grid_shape: Optional[Tuple[int, int]] = None,
                    methods: Optional[List[str]] = None) -> Dict[str, PerformanceMetrics]:
    """
    Benchmark multiple solvers on same problem.
    
    Parameters:
        matrix: Sparse matrix
        b: Right-hand side
        grid_shape: Grid dimensions (for multigrid)
        methods: List of methods to compare
        
    Returns:
        Dictionary mapping method names to PerformanceMetrics
    """
    if methods is None:
        methods = ['direct', 'gmres', 'bicgstab', 'multigrid']
    
    results = {}
    solver = OptimizedSolver(verbose=False)
    
    print(f"\nBenchmarking solver methods on {matrix.shape[0]}x{matrix.shape[0]} matrix ({matrix.nnz} nnz)")
    print("-" * 70)
    
    for method in methods:
        try:
            if method == 'direct':
                x, metric = solver.solve_direct(matrix, b)
                results[method] = metric
            elif method == 'gmres':
                x, _, metric = solver.solve_iterative_gmres(matrix, b)
                results[method] = metric
            elif method == 'bicgstab':
                x, _, metric = solver.solve_iterative_bicgstab(matrix, b)
                results[method] = metric
            elif method == 'multigrid':
                if grid_shape is None:
                    side = int(np.sqrt(matrix.shape[0]))
                    grid_shape = (side, side)
                x, _, metric = solver.solve_multigrid(matrix, b, grid_shape)
                results[method] = metric
            
            print(f"{method:15s}: {metric.time_elapsed:10.4f}s, "
                  f"residual={metric.residual:.3e}, "
                  f"{metric.throughput_gflops:6.2f} GFLOPS")
        except Exception as e:
            print(f"{method:15s}: FAILED - {str(e)}")
    
    print("-" * 70)
    
    return results


def estimate_memory_usage(matrix_shape: Tuple[int, int],
                         nnz: int,
                         n_rhs: int = 1,
                         include_workspace: bool = True) -> float:
    """
    Estimate memory usage for sparse solve.
    
    Parameters:
        matrix_shape: (rows, cols)
        nnz: Number of non-zeros
        n_rhs: Number of right-hand sides
        include_workspace: Include solver workspace
        
    Returns:
        Estimated memory in MB
    """
    # Matrix storage (CSR: values + colind + rowptr)
    matrix_mb = (nnz * 8 + nnz * 4 + (matrix_shape[0] + 1) * 4) / 1e6
    
    # Solution vectors
    solution_mb = (matrix_shape[0] * n_rhs * 8) / 1e6
    
    # RHS
    rhs_mb = (matrix_shape[0] * n_rhs * 8) / 1e6
    
    # Workspace (LU factorization, workspace for solvers)
    workspace_mb = 0.0
    if include_workspace:
        # Estimate workspace as 2x matrix + 2x solution
        workspace_mb = 2 * matrix_mb + 2 * solution_mb
    
    total_mb = matrix_mb + solution_mb + rhs_mb + workspace_mb
    
    return total_mb


def estimate_flops(matrix_size: int,
                  nnz: int,
                  n_iterations: int = 1) -> float:
    """
    Estimate FLOPs for sparse solve.
    
    Parameters:
        matrix_size: Size of matrix
        nnz: Number of non-zeros
        n_iterations: Number of iterations (for iterative solvers)
        
    Returns:
        Estimated FLOPs
    """
    # Matvec: 2*nnz (multiply + add per nonzero)
    # Preconditioner: ~4*matrix_size (diagonal solve)
    # Other operations: ~10*matrix_size
    
    flops_per_iter = 2 * nnz + 4 * matrix_size + 10 * matrix_size
    total_flops = flops_per_iter * max(1, n_iterations)
    
    return total_flops
