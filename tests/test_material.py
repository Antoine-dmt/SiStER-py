"""
Test suite for Material Grid Module

Tests:
    - MaterialGrid class instantiation and properties
    - Interpolation to normal and staggered nodes
    - Phase distribution handling
    - Property evaluation and ranges
    - Integration with Grid and ConfigurationManager
    - Performance benchmarks
"""

import pytest
import numpy as np
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from sister_py.material_grid import (
    MaterialGrid,
    MaterialProperties,
    MaterialGridMetadata,
    interpolate_to_normal_nodes,
    interpolate_to_staggered_nodes,
    create_layered_phase_distribution,
    create_two_phase_distribution,
)
from sister_py.grid import create_uniform_grid, create_zoned_grid
from sister_py.config import ConfigurationManager


class TestMaterialGridCreation:
    """Test MaterialGrid instantiation."""
    
    def test_material_grid_creation_simple(self):
        """Test creating simple uniform material grid."""
        grid = create_uniform_grid(0, 100e3, 0, 50e3, 11, 6)
        grid_dict = grid.to_dict()
        
        cfg_file = Path(__file__).parent.parent / "sister_py" / "data" / "defaults.yaml"
        if cfg_file.exists():
            cfg = ConfigurationManager.load(str(cfg_file))
            phase_array = np.ones((11, 6), dtype=int)
            
            matgrid = MaterialGrid(grid_dict, cfg, phase_array)
            
            assert matgrid.metadata.n_nodes_x == 11
            assert matgrid.metadata.n_nodes_y == 6
            assert matgrid.metadata.n_materials > 0
    
    def test_material_grid_repr(self):
        """Test MaterialGrid string representation."""
        grid = create_uniform_grid(0, 100e3, 0, 50e3, 11, 6)
        grid_dict = grid.to_dict()
        
        cfg_file = Path(__file__).parent.parent / "sister_py" / "data" / "defaults.yaml"
        if cfg_file.exists():
            cfg = ConfigurationManager.load(str(cfg_file))
            phase_array = np.ones((11, 6), dtype=int)
            matgrid = MaterialGrid(grid_dict, cfg, phase_array)
            
            repr_str = repr(matgrid)
            assert "MaterialGrid" in repr_str
            assert "nx=11" in repr_str
            assert "ny=6" in repr_str
    
    def test_material_grid_shape_mismatch(self):
        """Test error on shape mismatch."""
        grid = create_uniform_grid(0, 100e3, 0, 50e3, 11, 6)
        grid_dict = grid.to_dict()
        
        cfg_file = Path(__file__).parent.parent / "sister_py" / "data" / "defaults.yaml"
        if cfg_file.exists():
            cfg = ConfigurationManager.load(str(cfg_file))
            phase_array = np.ones((10, 5), dtype=int)  # Wrong shape!
            
            with pytest.raises(ValueError, match="phase_array_n shape"):
                MaterialGrid(grid_dict, cfg, phase_array)


class TestMaterialProperties:
    """Test material property computation."""
    
    def test_density_computation(self):
        """Test density computation on nodes."""
        grid = create_uniform_grid(0, 100e3, 0, 50e3, 11, 6)
        grid_dict = grid.to_dict()
        
        cfg_file = Path(__file__).parent.parent / "sister_py" / "data" / "defaults.yaml"
        if cfg_file.exists():
            cfg = ConfigurationManager.load(str(cfg_file))
            phase_array = np.ones((11, 6), dtype=int)
            matgrid = MaterialGrid(grid_dict, cfg, phase_array)
            
            # Check density values are reasonable (typical: 2700-3300 kg/m³)
            assert np.all(matgrid.density_n > 2000)
            assert np.all(matgrid.density_n < 4000)
    
    def test_viscosity_computation(self):
        """Test viscosity computation on nodes."""
        grid = create_uniform_grid(0, 100e3, 0, 50e3, 11, 6)
        grid_dict = grid.to_dict()
        
        cfg_file = Path(__file__).parent.parent / "sister_py" / "data" / "defaults.yaml"
        if cfg_file.exists():
            cfg = ConfigurationManager.load(str(cfg_file))
            phase_array = np.ones((11, 6), dtype=int)
            matgrid = MaterialGrid(grid_dict, cfg, phase_array)
            
            # Check viscosity is positive and in realistic range
            assert np.all(matgrid.viscosity_effective_n > 0)
            assert np.all(matgrid.viscosity_effective_n > 1e18)  # At least 1e18 Pa·s


class TestInterpolationToNormalNodes:
    """Test interpolation to normal nodes."""
    
    def test_interpolate_x_axis(self):
        """Test interpolation along x-axis."""
        values = np.array([[1, 2], [3, 4], [5, 6]])  # 3×2 array
        
        result = interpolate_to_normal_nodes(values, axis=0)
        
        expected = np.array([[2, 3], [4, 5]])  # (1+3)/2=2, (3+5)/2=4, etc.
        assert np.allclose(result, expected)
    
    def test_interpolate_y_axis(self):
        """Test interpolation along y-axis."""
        values = np.array([[1, 2, 3], [4, 5, 6]])  # 2×3 array
        
        result = interpolate_to_normal_nodes(values, axis=1)
        
        expected = np.array([[1.5, 2.5], [4.5, 5.5]])
        assert np.allclose(result, expected)
    
    def test_interpolate_invalid_axis(self):
        """Test error on invalid axis."""
        values = np.random.rand(3, 3)
        
        with pytest.raises(ValueError, match="axis must be"):
            interpolate_to_normal_nodes(values, axis=2)


class TestInterpolationToStaggeredNodes:
    """Test interpolation to staggered nodes."""
    
    def test_interpolate_x_staggered(self):
        """Test interpolation to x-staggered nodes."""
        values = np.array([[1, 2], [3, 4], [5, 6]])  # 3×2
        
        result = interpolate_to_staggered_nodes(values, axis=0)
        
        # X-staggered: average in x-direction
        expected = np.array([[2, 3], [4, 5]])  # 2×2
        assert result.shape == (2, 2)
        assert np.allclose(result, expected)
    
    def test_interpolate_y_staggered(self):
        """Test interpolation to y-staggered nodes."""
        values = np.array([[1, 2, 3], [4, 5, 6]])  # 2×3
        
        result = interpolate_to_staggered_nodes(values, axis=1)
        
        # Y-staggered: average in y-direction
        expected = np.array([[1.5, 2.5], [4.5, 5.5]])  # 2×2
        assert result.shape == (2, 2)
        assert np.allclose(result, expected)


class TestStaggeredNodeProperties:
    """Test staggered node property interpolation."""
    
    def test_density_on_x_staggered(self):
        """Test density interpolation to x-staggered nodes."""
        grid = create_uniform_grid(0, 100e3, 0, 50e3, 11, 6)
        grid_dict = grid.to_dict()
        
        cfg_file = Path(__file__).parent.parent / "sister_py" / "data" / "defaults.yaml"
        if cfg_file.exists():
            cfg = ConfigurationManager.load(str(cfg_file))
            phase_array = np.ones((11, 6), dtype=int)
            matgrid = MaterialGrid(grid_dict, cfg, phase_array)
            
            # X-staggered should have 10×6 (one less in x)
            assert matgrid.density_xs.shape == (10, 6)
            assert np.all(matgrid.density_xs > 0)
    
    def test_density_on_y_staggered(self):
        """Test density interpolation to y-staggered nodes."""
        grid = create_uniform_grid(0, 100e3, 0, 50e3, 11, 6)
        grid_dict = grid.to_dict()
        
        cfg_file = Path(__file__).parent.parent / "sister_py" / "data" / "defaults.yaml"
        if cfg_file.exists():
            cfg = ConfigurationManager.load(str(cfg_file))
            phase_array = np.ones((11, 6), dtype=int)
            matgrid = MaterialGrid(grid_dict, cfg, phase_array)
            
            # Y-staggered should have 11×5 (one less in y)
            assert matgrid.density_ys.shape == (11, 5)
            assert np.all(matgrid.density_ys > 0)
    
    def test_viscosity_staggered_interpolation(self):
        """Test viscosity interpolation to staggered nodes."""
        grid = create_uniform_grid(0, 100e3, 0, 50e3, 11, 6)
        grid_dict = grid.to_dict()
        
        cfg_file = Path(__file__).parent.parent / "sister_py" / "data" / "defaults.yaml"
        if cfg_file.exists():
            cfg = ConfigurationManager.load(str(cfg_file))
            phase_array = np.ones((11, 6), dtype=int)
            matgrid = MaterialGrid(grid_dict, cfg, phase_array)
            
            # Check interpolated viscosity in reasonable range
            assert np.all(matgrid.viscosity_xs > 0)
            assert np.all(matgrid.viscosity_ys > 0)


class TestPhaseDistribution:
    """Test phase distribution functions."""
    
    def test_layered_phase_distribution(self):
        """Test layered phase creation."""
        x = np.linspace(0, 100, 11)
        y = np.linspace(0, 100, 11)
        
        phase = create_layered_phase_distribution(
            x, y, 11, 11,
            layer_breaks=[0, 30, 100],
            layer_phases=[1, 2]
        )
        
        assert phase.shape == (11, 11)
        # Lower layer should be phase 1
        assert phase[0, 0] == 1
        # Upper layer should be phase 2
        assert phase[0, 10] == 2
    
    def test_two_phase_distribution(self):
        """Test two-phase layer over half-space."""
        x = np.linspace(0, 100, 11)
        y = np.linspace(0, 100, 11)
        
        phase = create_two_phase_distribution(
            x, y, 11, 11,
            transition_depth=50,
            phase1=1,
            phase2=2
        )
        
        assert phase.shape == (11, 11)
        # Below transition
        assert np.all(phase[:, :5] == 2)
        # Above transition
        assert np.all(phase[:, 6:] == 1)
    
    def test_layered_phase_error_wrong_length(self):
        """Test error on mismatched layer arrays."""
        x = np.linspace(0, 100, 11)
        y = np.linspace(0, 100, 11)
        
        with pytest.raises(ValueError, match="layer_breaks length"):
            create_layered_phase_distribution(
                x, y, 11, 11,
                layer_breaks=[0, 50, 100],
                layer_phases=[1]  # Should have 2 phases!
            )


class TestMaterialGridGeneration:
    """Test MaterialGrid.generate() class method."""
    
    def test_generate_with_default_phase(self):
        """Test generation with default uniform phase."""
        grid = create_uniform_grid(0, 100e3, 0, 50e3, 11, 6)
        grid_dict = grid.to_dict()
        
        cfg_file = Path(__file__).parent.parent / "sister_py" / "data" / "defaults.yaml"
        if cfg_file.exists():
            cfg = ConfigurationManager.load(str(cfg_file))
            
            matgrid = MaterialGrid.generate(cfg, grid_dict)
            
            assert matgrid.metadata.n_nodes_x == 11
            assert matgrid.metadata.n_nodes_y == 6
            assert np.all(matgrid.phase_n == 1)
    
    def test_generate_with_custom_phase_generator(self):
        """Test generation with custom phase function."""
        grid = create_uniform_grid(0, 100e3, 0, 50e3, 11, 6)
        grid_dict = grid.to_dict()
        
        cfg_file = Path(__file__).parent.parent / "sister_py" / "data" / "defaults.yaml"
        if cfg_file.exists():
            cfg = ConfigurationManager.load(str(cfg_file))
            
            # Custom generator: layer at y=25km
            def custom_phase(x, y, nx, ny):
                return create_two_phase_distribution(
                    x, y, nx, ny,
                    transition_depth=25e3,
                    phase1=1, phase2=2
                )
            
            matgrid = MaterialGrid.generate(cfg, grid_dict, custom_phase)
            
            # Should have both phases
            assert 1 in matgrid.phase_n
            assert 2 in matgrid.phase_n


class TestMaterialGridIndexing:
    """Test MaterialGrid indexing."""
    
    def test_getitem_density(self):
        """Test accessing density array."""
        grid = create_uniform_grid(0, 100e3, 0, 50e3, 11, 6)
        grid_dict = grid.to_dict()
        
        cfg_file = Path(__file__).parent.parent / "sister_py" / "data" / "defaults.yaml"
        if cfg_file.exists():
            cfg = ConfigurationManager.load(str(cfg_file))
            phase_array = np.ones((11, 6), dtype=int)
            matgrid = MaterialGrid(grid_dict, cfg, phase_array)
            
            density = matgrid['density_n']
            assert np.array_equal(density, matgrid.density_n)
    
    def test_getitem_viscosity(self):
        """Test accessing viscosity array."""
        grid = create_uniform_grid(0, 100e3, 0, 50e3, 11, 6)
        grid_dict = grid.to_dict()
        
        cfg_file = Path(__file__).parent.parent / "sister_py" / "data" / "defaults.yaml"
        if cfg_file.exists():
            cfg = ConfigurationManager.load(str(cfg_file))
            phase_array = np.ones((11, 6), dtype=int)
            matgrid = MaterialGrid(grid_dict, cfg, phase_array)
            
            visc = matgrid['viscosity_effective_n']
            assert np.array_equal(visc, matgrid.viscosity_effective_n)
    
    def test_getitem_invalid(self):
        """Test error on invalid key."""
        grid = create_uniform_grid(0, 100e3, 0, 50e3, 11, 6)
        grid_dict = grid.to_dict()
        
        cfg_file = Path(__file__).parent.parent / "sister_py" / "data" / "defaults.yaml"
        if cfg_file.exists():
            cfg = ConfigurationManager.load(str(cfg_file))
            phase_array = np.ones((11, 6), dtype=int)
            matgrid = MaterialGrid(grid_dict, cfg, phase_array)
            
            with pytest.raises(KeyError, match="Unknown property"):
                _ = matgrid['invalid_property']


class TestMaterialGridExport:
    """Test MaterialGrid.to_dict() export."""
    
    def test_export_to_dict(self):
        """Test exporting to dictionary."""
        grid = create_uniform_grid(0, 100e3, 0, 50e3, 11, 6)
        grid_dict = grid.to_dict()
        
        cfg_file = Path(__file__).parent.parent / "sister_py" / "data" / "defaults.yaml"
        if cfg_file.exists():
            cfg = ConfigurationManager.load(str(cfg_file))
            phase_array = np.ones((11, 6), dtype=int)
            matgrid = MaterialGrid(grid_dict, cfg, phase_array)
            
            export = matgrid.to_dict()
            
            assert 'grid_metadata' in export
            assert 'material_metadata' in export
            assert 'properties' in export
            assert 'phases' in export
            
            assert export['material_metadata']['n_nodes_x'] == 11
            assert export['material_metadata']['n_nodes_y'] == 6


class TestMaterialGridMetadata:
    """Test MaterialGridMetadata computation."""
    
    def test_metadata_viscosity_range(self):
        """Test viscosity range in metadata."""
        grid = create_uniform_grid(0, 100e3, 0, 50e3, 11, 6)
        grid_dict = grid.to_dict()
        
        cfg_file = Path(__file__).parent.parent / "sister_py" / "data" / "defaults.yaml"
        if cfg_file.exists():
            cfg = ConfigurationManager.load(str(cfg_file))
            phase_array = np.ones((11, 6), dtype=int)
            matgrid = MaterialGrid(grid_dict, cfg, phase_array)
            
            assert matgrid.metadata.min_viscosity > 0
            assert matgrid.metadata.max_viscosity >= matgrid.metadata.min_viscosity
    
    def test_metadata_density_range(self):
        """Test density range in metadata."""
        grid = create_uniform_grid(0, 100e3, 0, 50e3, 11, 6)
        grid_dict = grid.to_dict()
        
        cfg_file = Path(__file__).parent.parent / "sister_py" / "data" / "defaults.yaml"
        if cfg_file.exists():
            cfg = ConfigurationManager.load(str(cfg_file))
            phase_array = np.ones((11, 6), dtype=int)
            matgrid = MaterialGrid(grid_dict, cfg, phase_array)
            
            assert matgrid.metadata.min_density > 0
            assert matgrid.metadata.max_density >= matgrid.metadata.min_density


class TestMaterialGridPerformance:
    """Performance tests for material grid."""
    
    def test_performance_large_grid(self):
        """Test performance on large grid."""
        import time
        
        grid = create_uniform_grid(0, 1000e3, 0, 500e3, 201, 101)
        grid_dict = grid.to_dict()
        
        cfg_file = Path(__file__).parent.parent / "sister_py" / "data" / "defaults.yaml"
        if cfg_file.exists():
            cfg = ConfigurationManager.load(str(cfg_file))
            phase_array = np.ones((201, 101), dtype=int)
            
            start = time.time()
            matgrid = MaterialGrid(grid_dict, cfg, phase_array)
            elapsed = time.time() - start
            
            # Should complete in <150ms (vectorized computation)
            assert elapsed < 0.15
            assert matgrid is not None


class TestMaterialGridEdgeCases:
    """Test edge cases."""
    
    def test_minimal_grid(self):
        """Test minimal 2×2 grid."""
        grid = create_uniform_grid(0, 100, 0, 100, 2, 2)
        grid_dict = grid.to_dict()
        
        cfg_file = Path(__file__).parent.parent / "sister_py" / "data" / "defaults.yaml"
        if cfg_file.exists():
            cfg = ConfigurationManager.load(str(cfg_file))
            phase_array = np.ones((2, 2), dtype=int)
            
            matgrid = MaterialGrid(grid_dict, cfg, phase_array)
            
            assert matgrid.density_xs.shape == (1, 2)
            assert matgrid.density_ys.shape == (2, 1)
    
    def test_uniform_property_distribution(self):
        """Test uniform phase gives uniform properties."""
        grid = create_uniform_grid(0, 100e3, 0, 50e3, 21, 11)
        grid_dict = grid.to_dict()
        
        cfg_file = Path(__file__).parent.parent / "sister_py" / "data" / "defaults.yaml"
        if cfg_file.exists():
            cfg = ConfigurationManager.load(str(cfg_file))
            phase_array = np.ones((21, 11), dtype=int)
            
            matgrid = MaterialGrid(grid_dict, cfg, phase_array)
            
            # For uniform phase, all properties should be equal
            assert np.allclose(matgrid.density_n, matgrid.density_n[0, 0])
            assert np.allclose(matgrid.viscosity_effective_n, 
                             matgrid.viscosity_effective_n[0, 0])


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
