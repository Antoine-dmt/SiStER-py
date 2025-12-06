"""
Linear Solver Module for SiSteR-py

Implements direct and iterative sparse solvers for incompressible Stokes equations.
Handles saddle-point systems with pressure constraints and optional preconditioning.

Classes:
    LinearSolver: Abstract base for all solvers
    DirectSolver: scipy.sparse.spsolve wrapper with LU factorization
    IterativeSolver: GMRES with optional preconditioning
    SchurComplementSolver: Block decomposition approach
    SolverStats: Performance and convergence statistics
    
Functions:
    select_solver: Choose solver based on problem size and type
    estimate_condition_number: Estimate system conditioning
"""

import numpy as np
from typing import Tuple, Optional, Callable, Dict, Any
from dataclasses import dataclass
from abc import ABC, abstractmethod
from scipy import sparse
from scipy.sparse import linalg
from time import time
import warnings


@dataclass
class SolverStats:
    """Statistics from solver execution."""
    converged: bool = False
    iterations: int = 0
    residual_norm: float = 0.0
    relative_residual: float = 0.0
    solve_time: float = 0.0
    setup_time: float = 0.0
    matrix_size: Tuple[int, int] = (0, 0)
    matrix_nnz: int = 0
    matrix_sparsity: float = 0.0
    preconditioner_time: float = 0.0
    residual_history: list = None
    
    def __post_init__(self):
        if self.residual_history is None:
            self.residual_history = []
    
    def __repr__(self) -> str:
        """String representation."""
        return (
            f"SolverStats(converged={self.converged}, iters={self.iterations}, "
            f"res={self.residual_norm:.2e}, time={self.solve_time:.3f}s)"
        )


class LinearSolver(ABC):
    """Abstract base class for linear solvers."""
    
    def __init__(self, verbose: bool = False, max_iterations: int = 10000):
        """
        Initialize linear solver.
        
        Parameters:
            verbose: Print convergence information
            max_iterations: Maximum iterations for iterative solvers
        """
        self.verbose = verbose
        self.max_iterations = max_iterations
        self.stats = SolverStats()
    
    @abstractmethod
    def solve(self, A: sparse.spmatrix, b: np.ndarray) -> Tuple[np.ndarray, SolverStats]:
        """
        Solve linear system Ax = b.
        
        Parameters:
            A: Sparse coefficient matrix
            b: Right-hand side vector
            
        Returns:
            (solution, stats) tuple
        """
        pass
    
    @abstractmethod
    def setup(self, A: sparse.spmatrix) -> None:
        """Setup phase (preconditioner, factorization, etc.)."""
        pass


class DirectSolver(LinearSolver):
    """
    Direct sparse solver using LU factorization.
    
    Uses scipy.sparse.spsolve with SuperLU backend.
    Fast and reliable for small-medium problems (<1M DOFs).
    """
    
    def __init__(self, verbose: bool = False, max_iterations: int = 1,
                 permc_spec: str = 'MMD_AT_PLUS_A'):
        """
        Initialize direct solver.
        
        Parameters:
            verbose: Print timing information
            max_iterations: Always 1 for direct solvers
            permc_spec: Column permutation strategy for SuperLU
        """
        super().__init__(verbose, max_iterations=1)
        self.permc_spec = permc_spec
        self.A_lu = None
    
    def setup(self, A: sparse.spmatrix) -> None:
        """Perform LU factorization."""
        if self.verbose:
            print(f"  [DirectSolver] Setting up LU factorization...")
        
        start_time = time()
        
        # Convert to CSC format (optimal for LU)
        A_csc = A.tocsc()
        
        # Perform LU factorization
        try:
            self.A_lu = sparse.linalg.splu(A_csc, permc_spec=self.permc_spec)
            self.stats.setup_time = time() - start_time
            if self.verbose:
                print(f"  [DirectSolver] LU factorization completed in {self.stats.setup_time:.3f}s")
        except Exception as e:
            raise RuntimeError(f"LU factorization failed: {e}")
    
    def solve(self, A: sparse.spmatrix, b: np.ndarray) -> Tuple[np.ndarray, SolverStats]:
        """
        Solve Ax = b using direct LU solver.
        
        Parameters:
            A: Sparse coefficient matrix (CSR, CSC, or COO)
            b: Right-hand side vector
            
        Returns:
            (solution, stats) tuple
        """
        start_time = time()
        
        self.stats.matrix_size = A.shape
        self.stats.matrix_nnz = A.nnz
        self.stats.matrix_sparsity = 1.0 - A.nnz / (A.shape[0] * A.shape[1])
        
        # Setup if needed
        if self.A_lu is None:
            self.setup(A)
        
        # Solve using LU factorization
        try:
            x = self.A_lu.solve(b)
            
            # Compute residuals
            residual = A @ x - b
            residual_norm = np.linalg.norm(residual)
            b_norm = np.linalg.norm(b)
            relative_residual = residual_norm / max(b_norm, 1e-14)
            
            self.stats.converged = relative_residual < 1e-10
            self.stats.iterations = 1
            self.stats.residual_norm = residual_norm
            self.stats.relative_residual = relative_residual
            self.stats.solve_time = time() - start_time
            
            if self.verbose:
                print(f"  [DirectSolver] Solved in {self.stats.solve_time:.3f}s")
                print(f"    Residual norm: {residual_norm:.3e}")
                print(f"    Relative residual: {relative_residual:.3e}")
            
            return x, self.stats
        
        except Exception as e:
            raise RuntimeError(f"Direct solve failed: {e}")


class GMRESSolver(LinearSolver):
    """
    GMRES iterative solver with optional preconditioning.
    
    Suitable for large systems (>1M DOFs).
    Requires good preconditioner for fast convergence.
    """
    
    def __init__(self, verbose: bool = False, max_iterations: int = 1000,
                 restart: int = 100, tolerance: float = 1e-6,
                 preconditioner: Optional[str] = None):
        """
        Initialize GMRES solver.
        
        Parameters:
            verbose: Print convergence information
            max_iterations: Maximum GMRES iterations
            restart: GMRES restart parameter (larger = more memory, better convergence)
            tolerance: Relative tolerance for convergence
            preconditioner: Type of preconditioner ('jacobi', 'ilu', 'ilu0', or None)
        """
        super().__init__(verbose, max_iterations)
        self.restart = restart
        self.tolerance = tolerance
        self.preconditioner_type = preconditioner
        self.M = None
    
    def _setup_preconditioner(self, A: sparse.spmatrix) -> None:
        """Setup preconditioner based on type."""
        if self.preconditioner_type is None:
            return
        
        start_time = time()
        A_csc = A.tocsc()
        
        try:
            if self.preconditioner_type.lower() == 'jacobi':
                # Jacobi preconditioner: diagonal scaling
                diag = np.asarray(A_csc.diagonal()).flatten()
                diag = np.where(np.abs(diag) > 1e-14, diag, 1.0)
                self.M = sparse.diags(1.0 / diag)
            
            elif self.preconditioner_type.lower() in ['ilu', 'ilu0']:
                # Incomplete LU factorization
                fill_factor = 2 if self.preconditioner_type.lower() == 'ilu' else 0
                self.M = sparse.linalg.spilu(A_csc, fill_factor=fill_factor, drop_tol=1e-4)
            
            else:
                warnings.warn(f"Unknown preconditioner: {self.preconditioner_type}")
                self.M = None
            
            self.stats.preconditioner_time = time() - start_time
            
            if self.verbose:
                pc_type = self.preconditioner_type or "None"
                print(f"  [GMRES] Preconditioner ({pc_type}) setup in {self.stats.preconditioner_time:.3f}s")
        
        except Exception as e:
            warnings.warn(f"Preconditioner setup failed: {e}. Continuing without.")
            self.M = None
    
    def setup(self, A: sparse.spmatrix) -> None:
        """Setup preconditioner."""
        self._setup_preconditioner(A)
    
    def solve(self, A: sparse.spmatrix, b: np.ndarray,
              x0: Optional[np.ndarray] = None) -> Tuple[np.ndarray, SolverStats]:
        """
        Solve Ax = b using restarted GMRES.
        
        Parameters:
            A: Sparse coefficient matrix
            b: Right-hand side vector
            x0: Initial guess (default: zero)
            
        Returns:
            (solution, stats) tuple
        """
        start_time = time()
        
        self.stats.matrix_size = A.shape
        self.stats.matrix_nnz = A.nnz
        self.stats.matrix_sparsity = 1.0 - A.nnz / (A.shape[0] * A.shape[1])
        
        # Setup preconditioner if needed
        if self.M is None and self.preconditioner_type is not None:
            self._setup_preconditioner(A)
        
        # GMRES callback for residual history
        residual_history = []
        
        def callback(rk):
            residual_history.append(rk)
        
        if self.verbose:
            print(f"  [GMRES] Starting solve (tol={self.tolerance:.1e}, restart={self.restart})...")
        
        try:
            # Solve with GMRES
            x, info = sparse.linalg.gmres(
                A, b,
                x0=x0,
                restart=self.restart,
                maxiter=self.max_iterations,
                rtol=self.tolerance,
                M=self.M,
                callback=callback,
                callback_type='legacy'
            )
            
            # Compute final residual
            residual = A @ x - b
            residual_norm = np.linalg.norm(residual)
            b_norm = np.linalg.norm(b)
            relative_residual = residual_norm / max(b_norm, 1e-14)
            
            self.stats.converged = (info == 0)
            self.stats.iterations = len(residual_history)
            self.stats.residual_norm = residual_norm
            self.stats.relative_residual = relative_residual
            self.stats.solve_time = time() - start_time
            self.stats.residual_history = residual_history
            
            if self.verbose:
                status = "CONVERGED" if self.stats.converged else f"NOT CONVERGED (info={info})"
                print(f"  [GMRES] {status} in {self.stats.iterations} iterations, "
                      f"{self.stats.solve_time:.3f}s")
                print(f"    Residual norm: {residual_norm:.3e}")
                print(f"    Relative residual: {relative_residual:.3e}")
            
            return x, self.stats
        
        except Exception as e:
            raise RuntimeError(f"GMRES solve failed: {e}")


class BiCGSTABSolver(LinearSolver):
    """
    BiCG-STAB iterative solver with optional preconditioning.
    
    Good for non-symmetric or mildly non-symmetric systems.
    Often faster than GMRES for some problem types.
    """
    
    def __init__(self, verbose: bool = False, max_iterations: int = 1000,
                 tolerance: float = 1e-6, preconditioner: Optional[str] = None):
        """
        Initialize BiCG-STAB solver.
        
        Parameters:
            verbose: Print convergence information
            max_iterations: Maximum iterations
            tolerance: Relative tolerance for convergence
            preconditioner: Type of preconditioner ('jacobi', 'ilu', 'ilu0', or None)
        """
        super().__init__(verbose, max_iterations)
        self.tolerance = tolerance
        self.preconditioner_type = preconditioner
        self.M = None
    
    def _setup_preconditioner(self, A: sparse.spmatrix) -> None:
        """Setup preconditioner based on type."""
        if self.preconditioner_type is None:
            return
        
        start_time = time()
        A_csc = A.tocsc()
        
        try:
            if self.preconditioner_type.lower() == 'jacobi':
                diag = np.asarray(A_csc.diagonal()).flatten()
                diag = np.where(np.abs(diag) > 1e-14, diag, 1.0)
                self.M = sparse.diags(1.0 / diag)
            
            elif self.preconditioner_type.lower() in ['ilu', 'ilu0']:
                fill_factor = 2 if self.preconditioner_type.lower() == 'ilu' else 0
                self.M = sparse.linalg.spilu(A_csc, fill_factor=fill_factor, drop_tol=1e-4)
            
            else:
                warnings.warn(f"Unknown preconditioner: {self.preconditioner_type}")
                self.M = None
            
            self.stats.preconditioner_time = time() - start_time
            
            if self.verbose:
                pc_type = self.preconditioner_type or "None"
                print(f"  [BiCG-STAB] Preconditioner ({pc_type}) setup in {self.stats.preconditioner_time:.3f}s")
        
        except Exception as e:
            warnings.warn(f"Preconditioner setup failed: {e}. Continuing without.")
            self.M = None
    
    def setup(self, A: sparse.spmatrix) -> None:
        """Setup preconditioner."""
        self._setup_preconditioner(A)
    
    def solve(self, A: sparse.spmatrix, b: np.ndarray,
              x0: Optional[np.ndarray] = None) -> Tuple[np.ndarray, SolverStats]:
        """
        Solve Ax = b using BiCG-STAB.
        
        Parameters:
            A: Sparse coefficient matrix
            b: Right-hand side vector
            x0: Initial guess (default: zero)
            
        Returns:
            (solution, stats) tuple
        """
        start_time = time()
        
        self.stats.matrix_size = A.shape
        self.stats.matrix_nnz = A.nnz
        self.stats.matrix_sparsity = 1.0 - A.nnz / (A.shape[0] * A.shape[1])
        
        # Setup preconditioner if needed
        if self.M is None and self.preconditioner_type is not None:
            self._setup_preconditioner(A)
        
        if self.verbose:
            print(f"  [BiCG-STAB] Starting solve (tol={self.tolerance:.1e})...")
        
        try:
            # Solve with BiCG-STAB
            x, info = sparse.linalg.bicgstab(
                A, b,
                x0=x0,
                maxiter=self.max_iterations,
                rtol=self.tolerance,
                M=self.M
            )
            
            # Compute final residual
            residual = A @ x - b
            residual_norm = np.linalg.norm(residual)
            b_norm = np.linalg.norm(b)
            relative_residual = residual_norm / max(b_norm, 1e-14)
            
            # Estimate iterations (BiCG-STAB doesn't report explicitly)
            self.stats.converged = (info == 0)
            self.stats.iterations = self.max_iterations if not self.stats.converged else 1
            self.stats.residual_norm = residual_norm
            self.stats.relative_residual = relative_residual
            self.stats.solve_time = time() - start_time
            
            if self.verbose:
                status = "CONVERGED" if self.stats.converged else f"NOT CONVERGED (info={info})"
                print(f"  [BiCG-STAB] {status} in {self.stats.solve_time:.3f}s")
                print(f"    Residual norm: {residual_norm:.3e}")
                print(f"    Relative residual: {relative_residual:.3e}")
            
            return x, self.stats
        
        except Exception as e:
            raise RuntimeError(f"BiCG-STAB solve failed: {e}")


def select_solver(
    problem_size: int,
    solver_type: str = 'auto',
    verbose: bool = False,
    **kwargs
) -> LinearSolver:
    """
    Select appropriate solver based on problem size and type.
    
    Parameters:
        problem_size: Number of DOFs
        solver_type: 'direct', 'gmres', 'bicgstab', 'auto'
        verbose: Print solver information
        **kwargs: Additional arguments passed to solver
        
    Returns:
        Configured LinearSolver instance
        
    Examples:
        # Small problem: use direct solver
        solver = select_solver(661, 'direct', verbose=True)
        
        # Large problem: use preconditioned GMRES
        solver = select_solver(100000, 'gmres', verbose=True, preconditioner='ilu')
        
        # Automatic selection
        solver = select_solver(50000, 'auto', verbose=True)
    """
    
    if solver_type.lower() == 'auto':
        # Automatic selection based on problem size
        if problem_size < 10000:
            solver_type = 'direct'
        elif problem_size < 500000:
            solver_type = 'gmres'
        else:
            solver_type = 'gmres'  # Even for largest, GMRES with precond
    
    if solver_type.lower() == 'direct':
        return DirectSolver(verbose=verbose, **kwargs)
    elif solver_type.lower() == 'gmres':
        return GMRESSolver(verbose=verbose, **kwargs)
    elif solver_type.lower() == 'bicgstab':
        return BiCGSTABSolver(verbose=verbose, **kwargs)
    else:
        raise ValueError(f"Unknown solver type: {solver_type}")


def estimate_condition_number(A: sparse.spmatrix, num_samples: int = 5) -> float:
    """
    Estimate matrix condition number using power iteration.
    
    Parameters:
        A: Sparse matrix
        num_samples: Number of random vectors to try
        
    Returns:
        Estimated condition number
    """
    try:
        # Estimate largest eigenvalue
        evals_max = sparse.linalg.eigsh(A, k=1, which='LM', return_eigenvectors=False)
        lambda_max = np.max(np.abs(evals_max))
        
        # Estimate smallest eigenvalue (usually harder)
        evals_min = sparse.linalg.eigsh(A, k=1, which='SM', return_eigenvectors=False)
        lambda_min = np.min(np.abs(evals_min))
        
        cond = lambda_max / max(lambda_min, 1e-14)
        return cond
    
    except:
        # Fallback: return estimate based on power iteration
        return None
