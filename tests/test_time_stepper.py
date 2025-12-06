"""
Tests for Time Integration Module

Tests for:
- Forward Euler time stepping
- Backward Euler time stepping
- Adaptive time stepping
- Time step estimation
- Integration control
"""

import pytest
import numpy as np
from scipy import sparse

from sister_py.time_stepper import (
    TimeIntegrator, ForwardEulerIntegrator, BackwardEulerIntegrator,
    AdaptiveTimeStep, TimeStepper, TimeIntegrationStats,
    TimeStepScheme, TimeStepEstimate
)
from sister_py.grid import create_uniform_grid
from sister_py.material_grid import MaterialGrid
from sister_py.solver import SolverSystem, SolverConfig, BoundaryCondition, BCType
from sister_py.config import ConfigurationManager


class TestTimeIntegrationStats:
    """Tests for time integration statistics."""
    
    def test_stats_init(self):
        """Test TimeIntegrationStats initialization."""
        stats = TimeIntegrationStats()
        assert stats.step_number == 0
        assert stats.time == 0.0
        assert stats.dt == 0.0
        assert stats.solver_converged is False
    
    def test_stats_repr(self):
        """Test statistics string representation."""
        stats = TimeIntegrationStats(
            step_number=1,
            time=0.1,
            dt=0.01,
            velocity_max=1e-2,
            picard_iterations=3
        )
        repr_str = repr(stats)
        assert 'TimeStep 1' in repr_str
        assert 'e-02' in repr_str or '0.01' in repr_str


class TestForwardEulerIntegrator:
    """Tests for forward Euler time integration."""
    
    def test_forward_euler_init(self):
        """Test ForwardEulerIntegrator initialization."""
        integrator = ForwardEulerIntegrator(verbose=False)
        assert integrator.scheme == TimeStepScheme.FORWARD_EULER
        assert integrator.time == 0.0
        assert integrator.step == 0
        assert len(integrator.stats_history) == 0
    
    def test_forward_euler_scheme_type(self):
        """Test scheme type is correct."""
        integrator = ForwardEulerIntegrator()
        assert integrator.scheme == TimeStepScheme.FORWARD_EULER
        assert integrator.scheme.value == "forward_euler"


class TestBackwardEulerIntegrator:
    """Tests for backward Euler time integration."""
    
    def test_backward_euler_init(self):
        """Test BackwardEulerIntegrator initialization."""
        integrator = BackwardEulerIntegrator(verbose=False, max_newton_iter=10)
        assert integrator.scheme == TimeStepScheme.BACKWARD_EULER
        assert integrator.max_newton_iter == 10
    
    def test_backward_euler_scheme_type(self):
        """Test scheme type is correct."""
        integrator = BackwardEulerIntegrator()
        assert integrator.scheme == TimeStepScheme.BACKWARD_EULER


class TestAdaptiveTimeStep:
    """Tests for adaptive time stepping."""
    
    def test_adaptive_dt_init(self):
        """Test AdaptiveTimeStep initialization."""
        adaptive = AdaptiveTimeStep(
            cfl_number=0.5,
            dt_min=0.001,
            dt_max=0.1,
            dt_initial=0.01
        )
        assert adaptive.cfl == 0.5
        assert adaptive.dt_min == 0.001
        assert adaptive.dt_max == 0.1
        assert adaptive.dt_current == 0.01
    
    def test_time_step_estimate_init(self):
        """Test TimeStepEstimate initialization."""
        estimate = TimeStepEstimate(
            dt_cfl=0.01,
            dt_diffusion=0.02,
            dt_courant=0.015,
            dt_suggested=0.01
        )
        assert estimate.dt_cfl == 0.01
        assert estimate.dt_diffusion == 0.02
        assert estimate.safe_dt == min(0.01, 0.02, 0.015)
    
    def test_adjust_after_failure(self):
        """Test time step reduction after failure."""
        adaptive = AdaptiveTimeStep(dt_initial=0.1, adjust_factor=0.9)
        dt_new = adaptive.adjust_after_failure()
        
        assert dt_new < 0.1
        assert dt_new == pytest.approx(0.09)
    
    def test_adjust_after_success(self):
        """Test time step increase after success."""
        adaptive = AdaptiveTimeStep(dt_initial=0.1)
        dt_new = adaptive.adjust_after_success(acceleration_factor=1.1)
        
        assert dt_new > 0.1
        assert dt_new == pytest.approx(0.11)
    
    def test_dt_clipping(self):
        """Test that time step is clipped to min/max bounds."""
        adaptive = AdaptiveTimeStep(
            dt_min=0.001,
            dt_max=1.0,
            dt_initial=0.01
        )
        
        # Reduce below minimum
        for _ in range(20):
            adaptive.adjust_after_failure()
        assert adaptive.dt_current >= adaptive.dt_min
        
        # Increase above maximum
        adaptive.dt_current = 1.5
        adaptive.adjust_after_success(acceleration_factor=2.0)
        assert adaptive.dt_current <= adaptive.dt_max


class TestEstimateTimeStep:
    """Tests for time step estimation functions."""
    
    def test_estimate_dt_cfl(self):
        """Test CFL-based time step estimation."""
        from pathlib import Path
        # Create simple velocity field
        vx = 0.1 * np.ones((10, 10))  # Constant velocity
        vy = 0.0 * np.ones((11, 9))
        
        # Create simple grid
        grid = create_uniform_grid(0, 10, 0, 10, 11, 11)
        
        # Load configuration
        cfg_file = Path(__file__).parent.parent / "sister_py" / "data" / "defaults.yaml"
        cfg = ConfigurationManager.load(str(cfg_file))
        
        # Estimate time step
        adaptive = AdaptiveTimeStep(cfl_number=0.5)
        
        # Mock material grid
        phase_array = np.ones((11, 11), dtype=int)
        material_grid = MaterialGrid.generate(cfg, grid.to_dict())
        
        estimate = adaptive.estimate_dt(vx, vy, grid, material_grid)
        
        assert isinstance(estimate, TimeStepEstimate)
        assert estimate.dt_cfl > 0
        assert estimate.dt_diffusion > 0
        assert estimate.dt_suggested > 0
        assert estimate.constraint_active in ['CFL', 'diffusion', 'Courant']
    
    def test_estimate_dt_zero_velocity(self):
        """Test time step with zero velocity (diffusion-dominated)."""
        from pathlib import Path
        vx = np.zeros((10, 10))
        vy = np.zeros((11, 9))
        
        grid = create_uniform_grid(0, 10, 0, 10, 11, 11)
        cfg_file = Path(__file__).parent.parent / "sister_py" / "data" / "defaults.yaml"
        cfg = ConfigurationManager.load(str(cfg_file))
        material_grid = MaterialGrid.generate(cfg, grid.to_dict())
        
        adaptive = AdaptiveTimeStep()
        estimate = adaptive.estimate_dt(vx, vy, grid, material_grid)
        
        # With zero velocity, diffusion should dominate
        assert estimate.dt_cfl > estimate.dt_diffusion or estimate.constraint_active == 'diffusion'


class TestTimeStepper:
    """Tests for main time stepping controller."""
    
    def test_timestepper_init(self):
        """Test TimeStepper initialization."""
        from pathlib import Path
        # Create simple solver
        grid = create_uniform_grid(0, 10, 0, 10, 5, 5)
        cfg_file = Path(__file__).parent.parent / "sister_py" / "data" / "defaults.yaml"
        cfg = ConfigurationManager.load(str(cfg_file))
        material_grid = MaterialGrid.generate(cfg, grid.to_dict())
        
        bcs = [
            BoundaryCondition('left', BCType.VELOCITY, vx=0.0, vy=0.0),
            BoundaryCondition('right', BCType.VELOCITY, vx=0.0, vy=0.0),
            BoundaryCondition('top', BCType.FREE_SURFACE),
            BoundaryCondition('bottom', BCType.VELOCITY, vx=0.0, vy=0.0),
        ]
        
        solver = SolverSystem(grid, material_grid, cfg, bcs)
        
        # Create time stepper
        stepper = TimeStepper(solver, scheme=TimeStepScheme.FORWARD_EULER, verbose=False)
        
        assert isinstance(stepper.integrator, ForwardEulerIntegrator)
        assert isinstance(stepper.adaptive_dt, AdaptiveTimeStep)
        assert len(stepper.time_history) == 1
    
    def test_timestepper_backward_euler(self):
        """Test TimeStepper with backward Euler."""
        from pathlib import Path
        grid = create_uniform_grid(0, 10, 0, 10, 5, 5)
        cfg_file = Path(__file__).parent.parent / "sister_py" / "data" / "defaults.yaml"
        cfg = ConfigurationManager.load(str(cfg_file))
        material_grid = MaterialGrid.generate(cfg, grid.to_dict())
        
        bcs = [
            BoundaryCondition('left', BCType.VELOCITY, vx=0.0, vy=0.0),
            BoundaryCondition('right', BCType.VELOCITY, vx=0.0, vy=0.0),
            BoundaryCondition('top', BCType.FREE_SURFACE),
            BoundaryCondition('bottom', BCType.VELOCITY, vx=0.0, vy=0.0),
        ]
        
        solver = SolverSystem(grid, material_grid, cfg, bcs)
        stepper = TimeStepper(solver, scheme=TimeStepScheme.BACKWARD_EULER, verbose=False)
        
        assert isinstance(stepper.integrator, BackwardEulerIntegrator)


class TestTimeSchemeEnums:
    """Tests for time stepping scheme enumeration."""
    
    def test_scheme_enum_values(self):
        """Test scheme enumeration values."""
        assert TimeStepScheme.FORWARD_EULER.value == "forward_euler"
        assert TimeStepScheme.BACKWARD_EULER.value == "backward_euler"
        assert TimeStepScheme.RUNGE_KUTTA_2.value == "rk2"
        assert TimeStepScheme.PREDICTOR_CORRECTOR.value == "pc"
    
    def test_scheme_comparison(self):
        """Test scheme enumeration comparison."""
        assert TimeStepScheme.FORWARD_EULER != TimeStepScheme.BACKWARD_EULER
        assert TimeStepScheme.FORWARD_EULER == TimeStepScheme.FORWARD_EULER


class TestTimeStepperIntegration:
    """Integration tests for time stepping."""
    
    def test_time_history_tracking(self):
        """Test that time history is properly tracked."""
        from pathlib import Path
        grid = create_uniform_grid(0, 10, 0, 10, 5, 5)
        cfg_file = Path(__file__).parent.parent / "sister_py" / "data" / "defaults.yaml"
        cfg = ConfigurationManager.load(str(cfg_file))
        material_grid = MaterialGrid.generate(cfg, grid.to_dict())
        
        bcs = [
            BoundaryCondition('left', BCType.VELOCITY, vx=0.0, vy=0.0),
            BoundaryCondition('right', BCType.VELOCITY, vx=0.0, vy=0.0),
            BoundaryCondition('top', BCType.FREE_SURFACE),
            BoundaryCondition('bottom', BCType.VELOCITY, vx=0.0, vy=0.0),
        ]
        
        solver = SolverSystem(grid, material_grid, cfg, bcs)
        stepper = TimeStepper(solver, verbose=False)
        
        # Time history should start with t=0
        assert len(stepper.time_history) == 1
        assert stepper.time_history[0] == 0.0
    
    def test_adaptive_dt_properties(self):
        """Test adaptive time stepping properties."""
        from pathlib import Path
        grid = create_uniform_grid(0, 10, 0, 10, 5, 5)
        cfg_file = Path(__file__).parent.parent / "sister_py" / "data" / "defaults.yaml"
        cfg = ConfigurationManager.load(str(cfg_file))
        material_grid = MaterialGrid.generate(cfg, grid.to_dict())
        
        bcs = [
            BoundaryCondition('left', BCType.VELOCITY, vx=0.0, vy=0.0),
            BoundaryCondition('right', BCType.VELOCITY, vx=0.0, vy=0.0),
            BoundaryCondition('top', BCType.FREE_SURFACE),
            BoundaryCondition('bottom', BCType.VELOCITY, vx=0.0, vy=0.0),
        ]
        
        solver = SolverSystem(grid, material_grid, cfg, bcs)
        stepper = TimeStepper(solver, verbose=False)
        
        # Check adaptive dt properties
        assert stepper.adaptive_dt.dt_min > 0
        assert stepper.adaptive_dt.dt_max > stepper.adaptive_dt.dt_min


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
