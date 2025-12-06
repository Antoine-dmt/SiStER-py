"""
Test suite for Finite Difference Assembly Module

Tests:
    - FiniteDifferenceAssembler initialization
    - System matrix assembly
    - Boundary condition application
    - Matrix structure validation
"""

import pytest
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from sister_py.fd_assembly import FiniteDifferenceAssembler
from sister_py.grid import create_uniform_grid
from sister_py.material_grid import MaterialGrid
from sister_py.config import ConfigurationManager


class TestFiniteDifferenceAssemblerInit:
    """Test FD assembler initialization."""
    
    def test_assembler_creation(self):
        """Test creating assembler."""
        grid = create_uniform_grid(0, 100e3, 0, 50e3, 11, 6)
        grid_dict = grid.to_dict()
        
        cfg_file = Path(__file__).parent.parent / "sister_py" / "data" / "defaults.yaml"
        if cfg_file.exists():
            cfg = ConfigurationManager.load(str(cfg_file))
            phase_array = np.ones((11, 6), dtype=int)
            mat_grid = MaterialGrid(grid_dict, cfg, phase_array)
            
            assembler = FiniteDifferenceAssembler(grid, mat_grid)
            
            assert assembler.nx == 11
            assert assembler.ny == 6
            assert assembler.nx_s == 10
            assert assembler.ny_s == 5
    
    def test_assembler_dof_numbering(self):
        """Test DOF numbering scheme."""
        grid = create_uniform_grid(0, 100e3, 0, 50e3, 11, 6)
        grid_dict = grid.to_dict()
        
        cfg_file = Path(__file__).parent.parent / "sister_py" / "data" / "defaults.yaml"
        if cfg_file.exists():
            cfg = ConfigurationManager.load(str(cfg_file))
            phase_array = np.ones((11, 6), dtype=int)
            mat_grid = MaterialGrid(grid_dict, cfg, phase_array)
            
            assembler = FiniteDifferenceAssembler(grid, mat_grid)
            
            # Check DOF counts
            assert assembler.n_vx_dof == 10 * 6  # 60
            assert assembler.n_vy_dof == 11 * 5  # 55
            assert assembler.n_vel_dof == 60 + 55  # 115
            assert assembler.n_pres_dof == 11 * 6  # 66
            assert assembler.n_total_dof == 115 + 66  # 181
    
    def test_assembler_custom_body_force(self):
        """Test assembler with custom body force."""
        grid = create_uniform_grid(0, 100e3, 0, 50e3, 11, 6)
        grid_dict = grid.to_dict()
        
        cfg_file = Path(__file__).parent.parent / "sister_py" / "data" / "defaults.yaml"
        if cfg_file.exists():
            cfg = ConfigurationManager.load(str(cfg_file))
            phase_array = np.ones((11, 6), dtype=int)
            mat_grid = MaterialGrid(grid_dict, cfg, phase_array)
            
            gx = np.zeros((11, 6))
            gy = np.full((11, 6), -10.0)
            body_force = [gx, gy]
            
            assembler = FiniteDifferenceAssembler(grid, mat_grid, body_force)
            
            assert np.all(assembler.gx == 0)
            assert np.all(assembler.gy == -10.0)


class TestFiniteDifferenceAssembly:
    """Test system matrix assembly."""
    
    def test_assemble_simple_system(self):
        """Test assembling a simple system."""
        grid = create_uniform_grid(0, 100e3, 0, 50e3, 11, 6)
        grid_dict = grid.to_dict()
        
        cfg_file = Path(__file__).parent.parent / "sister_py" / "data" / "defaults.yaml"
        if cfg_file.exists():
            cfg = ConfigurationManager.load(str(cfg_file))
            phase_array = np.ones((11, 6), dtype=int)
            mat_grid = MaterialGrid(grid_dict, cfg, phase_array)
            
            assembler = FiniteDifferenceAssembler(grid, mat_grid)
            A, b = assembler.assemble_system()
            
            # Check matrix dimensions
            assert A.shape == (181, 181)
            assert b.shape == (181,)
            
            # Check that matrix is sparse
            nnz = A.nnz
            total_entries = 181 * 181
            sparsity = 1.0 - nnz / total_entries
            assert sparsity > 0.9  # Should be >90% sparse
    
    def test_matrix_symmetry_structure(self):
        """Test matrix structure properties."""
        grid = create_uniform_grid(0, 100e3, 0, 50e3, 11, 6)
        grid_dict = grid.to_dict()
        
        cfg_file = Path(__file__).parent.parent / "sister_py" / "data" / "defaults.yaml"
        if cfg_file.exists():
            cfg = ConfigurationManager.load(str(cfg_file))
            phase_array = np.ones((11, 6), dtype=int)
            mat_grid = MaterialGrid(grid_dict, cfg, phase_array)
            
            assembler = FiniteDifferenceAssembler(grid, mat_grid)
            A, b = assembler.assemble_system()
            
            # Saddle point matrix should have zero diagonal pressure block
            pres_start = assembler.n_vel_dof
            pres_end = assembler.n_total_dof
            pressure_diag = A.diagonal()[pres_start:pres_end]
            
            # Pressure diagonal should be zero (from continuity constraint)
            assert np.allclose(pressure_diag, 0.0, atol=1e-10)


class TestBoundaryConditionApplication:
    """Test boundary condition enforcement."""
    
    def test_velocity_bc_application(self):
        """Test applying velocity boundary conditions."""
        grid = create_uniform_grid(0, 100e3, 0, 50e3, 11, 6)
        grid_dict = grid.to_dict()
        
        cfg_file = Path(__file__).parent.parent / "sister_py" / "data" / "defaults.yaml"
        if cfg_file.exists():
            cfg = ConfigurationManager.load(str(cfg_file))
            phase_array = np.ones((11, 6), dtype=int)
            mat_grid = MaterialGrid(grid_dict, cfg, phase_array)
            
            assembler = FiniteDifferenceAssembler(grid, mat_grid)
            A, b = assembler.assemble_system()
            
            # Create boundary condition arrays
            bc_vx = np.full((10, 6), np.nan)  # No-slip on left and right
            bc_vx[0, :] = 0.0    # Left wall: vx = 0
            bc_vx[-1, :] = 0.1   # Right wall: vx = 0.1
            
            A_bc, b_bc = assembler.apply_velocity_bc(A, b, bc_vx=bc_vx)
            
            # Check that boundary rows have been replaced
            # Row for vx[0, :] should have 1 on diagonal
            assert A_bc[0, 0] == 1.0
            assert b_bc[0] == 0.0
            
            # Row for vx[-1, :] should have 1 on diagonal
            last_vx_row = (10-1) * 6 + 0
            assert A_bc[last_vx_row, last_vx_row] == 1.0
            assert b_bc[last_vx_row] == 0.1
    
    def test_mixed_bc_application(self):
        """Test applying mixed BCs to different velocity components."""
        grid = create_uniform_grid(0, 100e3, 0, 50e3, 11, 6)
        grid_dict = grid.to_dict()
        
        cfg_file = Path(__file__).parent.parent / "sister_py" / "data" / "defaults.yaml"
        if cfg_file.exists():
            cfg = ConfigurationManager.load(str(cfg_file))
            phase_array = np.ones((11, 6), dtype=int)
            mat_grid = MaterialGrid(grid_dict, cfg, phase_array)
            
            assembler = FiniteDifferenceAssembler(grid, mat_grid)
            A, b = assembler.assemble_system()
            
            # vx boundary conditions
            bc_vx = np.full((10, 6), np.nan)
            bc_vx[0, :] = 0.0
            
            # vy boundary conditions
            bc_vy = np.full((11, 5), np.nan)
            bc_vy[:, 0] = 0.0  # Bottom: no-slip
            bc_vy[:, -1] = 0.0 # Top: free surface approximated by zero
            
            A_bc, b_bc = assembler.apply_velocity_bc(A, b, bc_vx=bc_vx, bc_vy=bc_vy)
            
            # Check vx BCs applied
            assert A_bc[0, 0] == 1.0
            assert b_bc[0] == 0.0
            
            # Check vy BCs applied
            vy_start = assembler.n_vx_dof
            for i in range(11):
                # Bottom BC at vy[i, 0]
                row = vy_start + i * 5 + 0
                assert A_bc[row, row] == 1.0
                assert b_bc[row] == 0.0


class TestMatrixProperties:
    """Test mathematical properties of assembled matrix."""
    
    def test_matrix_real_entries(self):
        """Test that all matrix entries are real numbers."""
        grid = create_uniform_grid(0, 100e3, 0, 50e3, 11, 6)
        grid_dict = grid.to_dict()
        
        cfg_file = Path(__file__).parent.parent / "sister_py" / "data" / "defaults.yaml"
        if cfg_file.exists():
            cfg = ConfigurationManager.load(str(cfg_file))
            phase_array = np.ones((11, 6), dtype=int)
            mat_grid = MaterialGrid(grid_dict, cfg, phase_array)
            
            assembler = FiniteDifferenceAssembler(grid, mat_grid)
            A, b = assembler.assemble_system()
            
            # Check no NaN or Inf
            assert np.all(np.isfinite(A.data))
            assert np.all(np.isfinite(b))
    
    def test_rhs_nonzero_gravity(self):
        """Test that RHS contains gravity term."""
        grid = create_uniform_grid(0, 100e3, 0, 50e3, 11, 6)
        grid_dict = grid.to_dict()
        
        cfg_file = Path(__file__).parent.parent / "sister_py" / "data" / "defaults.yaml"
        if cfg_file.exists():
            cfg = ConfigurationManager.load(str(cfg_file))
            phase_array = np.ones((11, 6), dtype=int)
            mat_grid = MaterialGrid(grid_dict, cfg, phase_array)
            
            # Default body force has gravity
            assembler = FiniteDifferenceAssembler(grid, mat_grid)
            A, b = assembler.assemble_system()
            
            # RHS should have non-zero entries from gravity
            assert not np.allclose(b, 0.0)
