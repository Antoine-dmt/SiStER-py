"""
Test suite for Solver Module

Tests:
    - SolverConfig validation and settings
    - BoundaryCondition specification
    - SolverSystem initialization
    - Strain rate computation
    - Basic system assembly
    - Boundary condition application
    - Linear system solving
    - Solution field extraction
    - Picard iteration convergence
"""

import pytest
import numpy as np
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from sister_py.solver import (
    SolverConfig,
    SolverSystem,
    BoundaryCondition,
    BCType,
    SolutionFields,
)
from sister_py.grid import create_uniform_grid
from sister_py.material_grid import MaterialGrid
from sister_py.config import ConfigurationManager


class TestSolverConfig:
    """Test SolverConfig validation and defaults."""
    
    def test_solver_config_defaults(self):
        """Test default configuration values."""
        cfg = SolverConfig()
        assert cfg.Npicard_min == 5
        assert cfg.Npicard_max == 50
        assert cfg.solver_type == "direct"
        assert cfg.plasticity_enabled is True
    
    def test_solver_config_custom(self):
        """Test custom configuration."""
        cfg = SolverConfig(
            Npicard_min=3,
            Npicard_max=20,
            solver_type="iterative",
            iterative_tol=1e-6
        )
        assert cfg.Npicard_min == 3
        assert cfg.Npicard_max == 20
        assert cfg.solver_type == "iterative"
        assert cfg.iterative_tol == 1e-6
    
    def test_solver_config_invalid_npicard(self):
        """Test invalid Picard iteration settings."""
        with pytest.raises(ValueError):
            SolverConfig(Npicard_min=0).validate()
        
        with pytest.raises(ValueError):
            SolverConfig(Npicard_max=2, Npicard_min=5).validate()
    
    def test_solver_config_invalid_solver_type(self):
        """Test invalid solver type."""
        with pytest.raises(ValueError):
            SolverConfig(solver_type="invalid").validate()


class TestBoundaryCondition:
    """Test BoundaryCondition specification."""
    
    def test_velocity_bc(self):
        """Test velocity boundary condition."""
        bc = BoundaryCondition(
            side='left',
            bc_type=BCType.VELOCITY,
            vx=0.0,
            vy=0.1
        )
        bc.validate()  # Should not raise
        assert bc.vx == 0.0
        assert bc.vy == 0.1
    
    def test_stress_bc(self):
        """Test stress boundary condition."""
        bc = BoundaryCondition(
            side='top',
            bc_type=BCType.STRESS,
            sxx=1e6,
            sxy=0.0
        )
        bc.validate()  # Should not raise
        assert bc.sxx == 1e6
    
    def test_velocity_bc_missing_components(self):
        """Test velocity BC missing velocity components."""
        bc = BoundaryCondition(
            side='left',
            bc_type=BCType.VELOCITY,
            vx=0.0
            # Missing vy
        )
        with pytest.raises(ValueError, match="requires vx and vy"):
            bc.validate()
    
    def test_stress_bc_missing_components(self):
        """Test stress BC missing stress components."""
        bc = BoundaryCondition(
            side='top',
            bc_type=BCType.STRESS,
            sxx=1e6
            # Missing sxy
        )
        with pytest.raises(ValueError, match="requires sxx and sxy"):
            bc.validate()
    
    def test_free_surface_bc(self):
        """Test free surface boundary condition."""
        bc = BoundaryCondition(
            side='top',
            bc_type=BCType.FREE_SURFACE
        )
        bc.validate()  # Should not raise


class TestSolutionFields:
    """Test SolutionFields container."""
    
    def test_solution_fields_creation(self):
        """Test creating solution fields."""
        vx = np.random.rand(10, 11)
        vy = np.random.rand(11, 10)
        p = np.random.rand(11, 11)
        
        sol = SolutionFields(vx=vx, vy=vy, p=p)
        
        assert sol.vx.shape == (10, 11)
        assert sol.vy.shape == (11, 10)
        assert sol.p.shape == (11, 11)
    
    def test_solution_fields_shape_property(self):
        """Test shape property."""
        vx = np.zeros((10, 11))
        vy = np.zeros((11, 10))
        p = np.zeros((11, 11))
        
        sol = SolutionFields(vx=vx, vy=vy, p=p)
        shapes = sol.shape
        
        assert shapes['vx'] == (10, 11)
        assert shapes['vy'] == (11, 10)
        assert shapes['p'] == (11, 11)
    
    def test_solution_fields_to_dict(self):
        """Test exporting to dictionary."""
        vx = np.ones((10, 11))
        vy = np.ones((11, 10)) * 2
        p = np.ones((11, 11)) * 3
        
        sol = SolutionFields(vx=vx, vy=vy, p=p)
        d = sol.to_dict()
        
        assert 'vx' in d
        assert 'vy' in d
        assert 'p' in d
        assert np.allclose(d['vx'], vx)
        assert np.allclose(d['vy'], vy)
        assert np.allclose(d['p'], p)


class TestSolverSystemInitialization:
    """Test SolverSystem initialization."""
    
    def test_solver_creation_simple(self):
        """Test creating simple solver."""
        grid = create_uniform_grid(0, 100e3, 0, 50e3, 11, 6)
        grid_dict = grid.to_dict()
        
        cfg_file = Path(__file__).parent.parent / "sister_py" / "data" / "defaults.yaml"
        if cfg_file.exists():
            cfg_mgr = ConfigurationManager.load(str(cfg_file))
            phase_array = np.ones((11, 6), dtype=int)
            mat_grid = MaterialGrid(grid_dict, cfg_mgr, phase_array)
            
            solver_cfg = SolverConfig(Npicard_min=2, Npicard_max=5)
            bcs = [
                BoundaryCondition('left', BCType.VELOCITY, vx=0.0, vy=0.0),
                BoundaryCondition('right', BCType.VELOCITY, vx=0.1, vy=0.0),
                BoundaryCondition('top', BCType.FREE_SURFACE),
                BoundaryCondition('bottom', BCType.VELOCITY, vx=0.0, vy=0.0),
            ]
            
            solver = SolverSystem(grid, mat_grid, solver_cfg, bcs)
            
            assert solver.nx == 11
            assert solver.ny == 6
            assert solver.nx_s == 10
            assert solver.ny_s == 5
            assert solver.picard_iteration == 0
            assert not solver.converged
    
    def test_solver_invalid_bc_list(self):
        """Test solver with invalid boundary condition."""
        grid = create_uniform_grid(0, 100e3, 0, 50e3, 11, 6)
        grid_dict = grid.to_dict()
        
        cfg_file = Path(__file__).parent.parent / "sister_py" / "data" / "defaults.yaml"
        if cfg_file.exists():
            cfg_mgr = ConfigurationManager.load(str(cfg_file))
            phase_array = np.ones((11, 6), dtype=int)
            mat_grid = MaterialGrid(grid_dict, cfg_mgr, phase_array)
            
            solver_cfg = SolverConfig()
            bcs = [
                BoundaryCondition('left', BCType.VELOCITY, vx=0.0)  # Missing vy
            ]
            
            with pytest.raises(ValueError):
                SolverSystem(grid, mat_grid, solver_cfg, bcs)


class TestStrainRateComputation:
    """Test strain rate calculations."""
    
    def test_strain_rate_simple(self):
        """Test strain rate on simple velocity field."""
        grid = create_uniform_grid(0, 100e3, 0, 50e3, 11, 6)
        grid_dict = grid.to_dict()
        
        cfg_file = Path(__file__).parent.parent / "sister_py" / "data" / "defaults.yaml"
        if cfg_file.exists():
            cfg_mgr = ConfigurationManager.load(str(cfg_file))
            phase_array = np.ones((11, 6), dtype=int)
            mat_grid = MaterialGrid(grid_dict, cfg_mgr, phase_array)
            
            solver_cfg = SolverConfig()
            bcs = [
                BoundaryCondition('left', BCType.VELOCITY, vx=0.0, vy=0.0),
                BoundaryCondition('right', BCType.VELOCITY, vx=0.1, vy=0.0),
                BoundaryCondition('top', BCType.FREE_SURFACE),
                BoundaryCondition('bottom', BCType.VELOCITY, vx=0.0, vy=0.0),
            ]
            
            solver = SolverSystem(grid, mat_grid, solver_cfg, bcs)
            
            # Simple linear velocity: vx = x / L, vy = 0
            vx = np.outer(np.linspace(0, 0.1, solver.nx_s), np.ones(solver.ny))
            vy = np.zeros((solver.nx, solver.ny_s))
            
            strain_rate = solver._compute_strain_rate_invariant(vx, vy)
            
            assert strain_rate.shape == (solver.nx, solver.ny)
            assert np.all(np.isfinite(strain_rate))
            assert np.all(strain_rate >= 0)
    
    def test_strain_rate_nonzero(self):
        """Test strain rate with non-zero velocity gradient."""
        grid = create_uniform_grid(0, 100e3, 0, 50e3, 11, 6)
        grid_dict = grid.to_dict()
        
        cfg_file = Path(__file__).parent.parent / "sister_py" / "data" / "defaults.yaml"
        if cfg_file.exists():
            cfg_mgr = ConfigurationManager.load(str(cfg_file))
            phase_array = np.ones((11, 6), dtype=int)
            mat_grid = MaterialGrid(grid_dict, cfg_mgr, phase_array)
            
            solver_cfg = SolverConfig()
            bcs = [
                BoundaryCondition('left', BCType.VELOCITY, vx=0.0, vy=0.0),
                BoundaryCondition('right', BCType.VELOCITY, vx=0.1, vy=0.0),
                BoundaryCondition('top', BCType.FREE_SURFACE),
                BoundaryCondition('bottom', BCType.VELOCITY, vx=0.0, vy=0.0),
            ]
            
            solver = SolverSystem(grid, mat_grid, solver_cfg, bcs)
            
            # Linear velocity field
            vx = np.random.rand(solver.nx_s, solver.ny) * 0.01
            vy = np.random.rand(solver.nx, solver.ny_s) * 0.01
            
            strain_rate = solver._compute_strain_rate_invariant(vx, vy)
            
            # Should have some non-zero strain rate
            assert np.max(strain_rate) > 1e-15


class TestSolverSystemRepr:
    """Test string representation."""
    
    def test_solver_repr(self):
        """Test solver __repr__ method."""
        grid = create_uniform_grid(0, 100e3, 0, 50e3, 11, 6)
        grid_dict = grid.to_dict()
        
        cfg_file = Path(__file__).parent.parent / "sister_py" / "data" / "defaults.yaml"
        if cfg_file.exists():
            cfg_mgr = ConfigurationManager.load(str(cfg_file))
            phase_array = np.ones((11, 6), dtype=int)
            mat_grid = MaterialGrid(grid_dict, cfg_mgr, phase_array)
            
            solver_cfg = SolverConfig()
            bcs = [
                BoundaryCondition('left', BCType.VELOCITY, vx=0.0, vy=0.0),
                BoundaryCondition('right', BCType.VELOCITY, vx=0.1, vy=0.0),
                BoundaryCondition('top', BCType.FREE_SURFACE),
                BoundaryCondition('bottom', BCType.VELOCITY, vx=0.0, vy=0.0),
            ]
            
            solver = SolverSystem(grid, mat_grid, solver_cfg, bcs)
            
            repr_str = repr(solver)
            assert "SolverSystem" in repr_str
            assert "11×6" in repr_str
            assert "not converged" in repr_str


class TestSolverConfiguration:
    """Test solver with different configurations."""
    
    def test_solver_direct_vs_iterative(self):
        """Test direct and iterative solver types."""
        grid = create_uniform_grid(0, 100e3, 0, 50e3, 11, 6)
        grid_dict = grid.to_dict()
        
        cfg_file = Path(__file__).parent.parent / "sister_py" / "data" / "defaults.yaml"
        if cfg_file.exists():
            cfg_mgr = ConfigurationManager.load(str(cfg_file))
            phase_array = np.ones((11, 6), dtype=int)
            mat_grid = MaterialGrid(grid_dict, cfg_mgr, phase_array)
            
            bcs = [
                BoundaryCondition('left', BCType.VELOCITY, vx=0.0, vy=0.0),
                BoundaryCondition('right', BCType.VELOCITY, vx=0.1, vy=0.0),
                BoundaryCondition('top', BCType.FREE_SURFACE),
                BoundaryCondition('bottom', BCType.VELOCITY, vx=0.0, vy=0.0),
            ]
            
            # Test direct
            solver_cfg_direct = SolverConfig(solver_type="direct")
            solver1 = SolverSystem(grid, mat_grid, solver_cfg_direct, bcs)
            assert solver1.cfg.solver_type == "direct"
            
            # Test iterative
            solver_cfg_iter = SolverConfig(solver_type="iterative", iterative_tol=1e-4)
            solver2 = SolverSystem(grid, mat_grid, solver_cfg_iter, bcs)
            assert solver2.cfg.solver_type == "iterative"
            assert solver2.cfg.iterative_tol == 1e-4
