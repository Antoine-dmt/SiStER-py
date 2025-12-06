"""
Time Integration Module for SiSteR-py

Implements time stepping schemes for geodynamic simulations.
Handles forward Euler, backward Euler (implicit), and adaptive time stepping.

Classes:
    TimeIntegrator: Abstract base for time stepping schemes
    ForwardEulerIntegrator: Explicit forward Euler scheme
    BackwardEulerIntegrator: Implicit backward Euler scheme (stable, expensive)
    AdaptiveTimeStep: Adaptive time stepping with CFL constraint
    TimeStepper: Main time integration controller
    
Functions:
    estimate_dt_cfl: Estimate time step from CFL condition
    estimate_dt_diffusion: Estimate time step from diffusion stability
"""

import numpy as np
from typing import Tuple, Optional, Callable, Dict, Any, List
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from enum import Enum
import warnings


class TimeStepScheme(Enum):
    """Time stepping scheme type."""
    FORWARD_EULER = "forward_euler"      # Explicit, fast, conditionally stable
    BACKWARD_EULER = "backward_euler"    # Implicit, slow, unconditionally stable
    RUNGE_KUTTA_2 = "rk2"                # Explicit RK2, moderate stability
    PREDICTOR_CORRECTOR = "pc"           # Implicit with predictor-corrector


@dataclass
class TimeStepEstimate:
    """Time step size estimate with constraints."""
    dt_cfl: float              # CFL-based time step
    dt_diffusion: float        # Diffusion stability time step
    dt_courant: float          # Courant condition (advection)
    dt_suggested: float        # Recommended safe time step
    constraint_active: str = "none"     # Which constraint is active
    
    @property
    def safe_dt(self) -> float:
        """Return safe time step (most restrictive)."""
        return np.min([self.dt_cfl, self.dt_diffusion, self.dt_courant])


@dataclass
class TimeIntegrationStats:
    """Statistics from time integration step."""
    step_number: int = 0
    time: float = 0.0
    dt: float = 0.0
    scheme: str = "unknown"
    solver_iterations: int = 0
    solver_converged: bool = False
    picard_iterations: int = 0
    picard_converged: bool = False
    velocity_max: float = 0.0
    strain_rate_max: float = 0.0
    stress_max: float = 0.0
    pressure_rms: float = 0.0
    cpu_time: float = 0.0
    
    def __repr__(self) -> str:
        """String representation."""
        return (
            f"TimeStep {self.step_number}: t={self.time:.2e}, dt={self.dt:.2e}, "
            f"v_max={self.velocity_max:.2e}, picard={self.picard_iterations}"
        )


class TimeIntegrator(ABC):
    """Abstract base class for time integration schemes."""
    
    def __init__(self, scheme: TimeStepScheme, verbose: bool = False):
        """
        Initialize time integrator.
        
        Parameters:
            scheme: Time stepping scheme type
            verbose: Print convergence information
        """
        self.scheme = scheme
        self.verbose = verbose
        self.time = 0.0
        self.step = 0
        self.stats_history: List[TimeIntegrationStats] = []
    
    @abstractmethod
    def step(self, 
             solver,
             material_grid,
             dt: float) -> Tuple[float, TimeIntegrationStats]:
        """
        Perform one time step.
        
        Parameters:
            solver: SolverSystem instance
            material_grid: Material grid for property updates
            dt: Time step size
            
        Returns:
            (new_time, stats) tuple
        """
        pass
    
    def add_stats(self, stats: TimeIntegrationStats) -> None:
        """Record time step statistics."""
        self.stats_history.append(stats)


class ForwardEulerIntegrator(TimeIntegrator):
    """
    Forward Euler time integration (explicit).
    
    Scheme: u^{n+1} = u^n + dt * f(u^n)
    
    Advantages:
        - Simple, no implicit solve needed
        - Fast per step
    
    Disadvantages:
        - Conditionally stable (CFL constraint)
        - Explicit only works for diffusion if dt very small
    """
    
    def __init__(self, verbose: bool = False):
        """Initialize forward Euler integrator."""
        super().__init__(TimeStepScheme.FORWARD_EULER, verbose)
    
    def step(self,
             solver,
             material_grid,
             dt: float) -> Tuple[float, TimeIntegrationStats]:
        """
        Perform forward Euler time step: u^{n+1} = u^n + dt * du/dt
        
        Parameters:
            solver: SolverSystem instance
            material_grid: Current material properties
            dt: Time step size
            
        Returns:
            (new_time, stats) tuple
        """
        from time import time as timer
        step_start = timer()
        
        # Current solution
        u = solver.solution
        vx_n = u.vx.copy()
        vy_n = u.vy.copy()
        
        # Compute time derivatives (advection, stress update)
        dvx_dt = self._compute_velocity_rhs(vx_n, vy_n, material_grid)
        dvy_dt = self._compute_velocity_rhs_y(vx_n, vy_n, material_grid)
        
        # Update: u^{n+1} = u^n + dt * du/dt
        vx_new = vx_n + dt * dvx_dt
        vy_new = vy_n + dt * dvy_dt
        
        # Solve Stokes to get pressure for this velocity
        p_new = solver._compute_pressure(vx_new, vy_new)
        
        # Update solver solution
        from sister_py.solver import SolutionFields
        solver.solution = SolutionFields(vx=vx_new, vy=vy_new, p=p_new)
        
        # Compute statistics
        stats = TimeIntegrationStats(
            step_number=self.step,
            time=self.time + dt,
            dt=dt,
            scheme=self.scheme.value,
            velocity_max=np.max(np.abs(vx_new)),
            solver_converged=True,
            picard_iterations=0,
            picard_converged=True,
            cpu_time=timer() - step_start
        )
        
        self.step += 1
        self.time += dt
        self.add_stats(stats)
        
        if self.verbose:
            print(f"  [ForwardEuler] Step {self.step}: t={self.time:.2e}, v_max={stats.velocity_max:.2e}")
        
        return self.time, stats
    
    def _compute_velocity_rhs(self, vx, vy, material_grid):
        """Compute RHS for x-velocity advection-diffusion."""
        # Placeholder: would implement full advection-diffusion
        return np.zeros_like(vx)
    
    def _compute_velocity_rhs_y(self, vx, vy, material_grid):
        """Compute RHS for y-velocity advection-diffusion."""
        # Placeholder: would implement full advection-diffusion
        return np.zeros_like(vy)


class BackwardEulerIntegrator(TimeIntegrator):
    """
    Backward Euler time integration (implicit).
    
    Scheme: u^{n+1} = u^n + dt * f(u^{n+1})
    
    Advantages:
        - Unconditionally stable (no CFL constraint)
        - Good for stiff problems
    
    Disadvantages:
        - Requires implicit solve (expensive)
        - Higher computational cost per step
    """
    
    def __init__(self, verbose: bool = False, max_newton_iter: int = 5):
        """
        Initialize backward Euler integrator.
        
        Parameters:
            verbose: Print convergence information
            max_newton_iter: Maximum Newton iterations for implicit solve
        """
        super().__init__(TimeStepScheme.BACKWARD_EULER, verbose)
        self.max_newton_iter = max_newton_iter
    
    def step(self,
             solver,
             material_grid,
             dt: float) -> Tuple[float, TimeIntegrationStats]:
        """
        Perform backward Euler time step using Newton iteration.
        
        Parameters:
            solver: SolverSystem instance
            material_grid: Current material properties
            dt: Time step size
            
        Returns:
            (new_time, stats) tuple
        """
        from time import time as timer
        step_start = timer()
        
        # Initial guess: use solution from previous step
        u_old = solver.solution
        vx_guess = u_old.vx.copy()
        vy_guess = u_old.vy.copy()
        
        # Newton iteration for implicit solve
        for newton_iter in range(self.max_newton_iter):
            # Solve for u^{n+1}  using current guess
            # F(u^{n+1}) = 0 where F = (u^{n+1} - u^n)/dt - RHS(u^{n+1})
            
            # Evaluate RHS at guess
            rhs_x = self._compute_velocity_rhs(vx_guess, vy_guess, material_grid)
            rhs_y = self._compute_velocity_rhs_y(vx_guess, vy_guess, material_grid)
            
            # Update guess
            vx_new = u_old.vx + dt * rhs_x
            vy_new = u_old.vy + dt * rhs_y
            
            # Check convergence
            residual = np.max([
                np.max(np.abs(vx_new - vx_guess)),
                np.max(np.abs(vy_new - vy_guess))
            ])
            
            vx_guess = vx_new
            vy_guess = vy_new
            
            if self.verbose and newton_iter < 3:
                print(f"    Newton iteration {newton_iter+1}: residual={residual:.2e}")
            
            if residual < 1e-6:
                break
        
        # Solve Stokes to get pressure
        p_new = solver._compute_pressure(vx_new, vy_new)
        
        # Update solver solution
        from sister_py.solver import SolutionFields
        solver.solution = SolutionFields(vx=vx_new, vy=vy_new, p=p_new)
        
        # Compute statistics
        stats = TimeIntegrationStats(
            step_number=self.step,
            time=self.time + dt,
            dt=dt,
            scheme=self.scheme.value,
            velocity_max=np.max(np.abs(vx_new)),
            solver_converged=True,
            picard_iterations=newton_iter + 1,
            picard_converged=residual < 1e-6,
            cpu_time=timer() - step_start
        )
        
        self.step += 1
        self.time += dt
        self.add_stats(stats)
        
        if self.verbose:
            print(f"  [BackwardEuler] Step {self.step}: t={self.time:.2e}, "
                  f"v_max={stats.velocity_max:.2e}, newton_iters={newton_iter+1}")
        
        return self.time, stats
    
    def _compute_velocity_rhs(self, vx, vy, material_grid):
        """Compute RHS for x-velocity."""
        return np.zeros_like(vx)
    
    def _compute_velocity_rhs_y(self, vx, vy, material_grid):
        """Compute RHS for y-velocity."""
        return np.zeros_like(vy)


class AdaptiveTimeStep:
    """Adaptive time stepping controller with CFL and stability constraints."""
    
    def __init__(self, 
                 cfl_number: float = 0.5,
                 dt_min: float = 1e-3,
                 dt_max: float = 1e3,
                 dt_initial: float = 1.0,
                 adjust_factor: float = 0.9):
        """
        Initialize adaptive time stepping.
        
        Parameters:
            cfl_number: CFL coefficient (typical: 0.5)
            dt_min: Minimum allowed time step
            dt_max: Maximum allowed time step
            dt_initial: Initial time step suggestion
            adjust_factor: Factor to reduce dt after failed step
        """
        self.cfl = cfl_number
        self.dt_min = dt_min
        self.dt_max = dt_max
        self.dt_current = dt_initial
        self.adjust_factor = adjust_factor
    
    def estimate_dt(self,
                    vx: np.ndarray,
                    vy: np.ndarray,
                    grid,
                    material_grid) -> TimeStepEstimate:
        """
        Estimate appropriate time step from CFL and stability conditions.
        
        Parameters:
            vx: X-velocity on x-staggered nodes
            vy: Y-velocity on y-staggered nodes
            grid: Computational grid
            material_grid: Material properties
            
        Returns:
            TimeStepEstimate with suggested dt
        """
        # Grid spacing
        dx = np.min(np.diff(grid.x_n))
        dy = np.min(np.diff(grid.y_n))
        
        # 1. CFL condition for advection: dt < CFL * h / |v|
        v_max = np.max(np.abs([vx.max(), vx.min(), vy.max(), vy.min()]))
        v_max = max(v_max, 1e-16)  # Avoid division by zero
        dt_cfl = self.cfl * min(dx, dy) / v_max
        
        # 2. Diffusion stability: dt < 0.25 * h^2 / κ
        # κ = k / (ρ * cp) is thermal diffusivity
        kappa = 1e-6  # Typical crustal diffusivity
        dt_diffusion = 0.25 * min(dx, dy)**2 / max(kappa, 1e-16)
        
        # 3. Courant number for advection
        dt_courant = 0.25 * min(dx, dy) / max(v_max, 1e-16)
        
        # Most restrictive constraint
        dt_safe = np.min([dt_cfl, dt_diffusion, dt_courant])
        dt_safe = np.clip(dt_safe, self.dt_min, self.dt_max)
        
        # Identify active constraint
        if dt_safe == dt_cfl:
            constraint = "CFL"
        elif dt_safe == dt_diffusion:
            constraint = "diffusion"
        else:
            constraint = "Courant"
        
        estimate = TimeStepEstimate(
            dt_cfl=dt_cfl,
            dt_diffusion=dt_diffusion,
            dt_courant=dt_courant,
            dt_suggested=dt_safe,
            constraint_active=constraint
        )
        
        self.dt_current = dt_safe
        return estimate
    
    def adjust_after_failure(self) -> float:
        """Reduce time step after solver failure."""
        self.dt_current *= self.adjust_factor
        self.dt_current = max(self.dt_current, self.dt_min)
        return self.dt_current
    
    def adjust_after_success(self, acceleration_factor: float = 1.05) -> float:
        """Increase time step after successful step."""
        self.dt_current *= acceleration_factor
        self.dt_current = min(self.dt_current, self.dt_max)
        return self.dt_current


class TimeStepper:
    """
    Main time integration controller for geodynamic simulations.
    
    Manages:
    - Time stepping scheme selection
    - Adaptive time stepping
    - Solution output and checkpointing
    - Simulation statistics
    """
    
    def __init__(self,
                 solver,
                 scheme: TimeStepScheme = TimeStepScheme.FORWARD_EULER,
                 verbose: bool = True):
        """
        Initialize time stepper.
        
        Parameters:
            solver: SolverSystem instance
            scheme: Time stepping scheme
            verbose: Print progress information
        """
        self.solver = solver
        self.verbose = verbose
        
        # Select integrator
        if scheme == TimeStepScheme.FORWARD_EULER:
            self.integrator = ForwardEulerIntegrator(verbose=verbose)
        elif scheme == TimeStepScheme.BACKWARD_EULER:
            self.integrator = BackwardEulerIntegrator(verbose=verbose)
        else:
            raise ValueError(f"Unknown time stepping scheme: {scheme}")
        
        # Adaptive time stepping
        self.adaptive_dt = AdaptiveTimeStep()
        
        # Statistics
        self.time_history: List[float] = [0.0]
        self.energy_history: List[float] = []
    
    def integrate(self,
                  n_steps: int,
                  dt_init: float = 1.0,
                  t_max: Optional[float] = None) -> Dict[str, Any]:
        """
        Run time integration for specified number of steps.
        
        Parameters:
            n_steps: Number of time steps to perform
            dt_init: Initial time step size
            t_max: Maximum simulation time (stops if reached)
            
        Returns:
            Dictionary with integration results and statistics
        """
        if self.verbose:
            print(f"\nStarting time integration: {n_steps} steps, scheme={self.integrator.scheme.value}")
        
        for step in range(n_steps):
            # Estimate adaptive time step
            dt_est = self.adaptive_dt.estimate_dt(
                self.solver.solution.vx,
                self.solver.solution.vy,
                self.solver.grid,
                self.solver.material_grid
            )
            dt = dt_est.dt_suggested if step > 0 else dt_init
            
            if self.verbose:
                print(f"Step {step+1}/{n_steps}: dt={dt:.2e} ({dt_est.constraint_active})")
            
            # Check time limit
            if t_max is not None and self.integrator.time + dt > t_max:
                dt = t_max - self.integrator.time
                if dt < self.adaptive_dt.dt_min:
                    if self.verbose:
                        print(f"Reached t_max = {t_max}")
                    break
            
            try:
                # Perform time step
                new_time, stats = self.integrator.step(
                    self.solver,
                    self.solver.material_grid,
                    dt
                )
                
                self.time_history.append(new_time)
                
                # Adapt time step for next iteration
                self.adaptive_dt.adjust_after_success()
                
            except Exception as e:
                warnings.warn(f"Time step failed: {e}. Reducing dt.")
                self.adaptive_dt.adjust_after_failure()
                continue
        
        # Summary
        result = {
            'n_steps_completed': len(self.time_history) - 1,
            'time_final': self.integrator.time,
            'time_history': self.time_history,
            'stats_history': self.integrator.stats_history,
        }
        
        if self.verbose:
            print(f"\nIntegration complete: t_final={self.integrator.time:.2e}, "
                  f"steps={len(self.time_history)-1}")
        
        return result
