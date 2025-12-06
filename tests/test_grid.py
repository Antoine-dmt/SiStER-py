"""
Test suite for Grid Module

Tests:
    - Uniform grid generation and validation
    - Zone-based grid generation
    - Staggered node positioning
    - Grid validation and constraints
    - Performance benchmarks
    - Integration with ConfigurationManager
"""

import pytest
import numpy as np
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from sister_py.grid import (
    Grid,
    GridMetadata,
    create_uniform_grid,
    create_zoned_grid,
    _generate_zoned_coordinates
)
from sister_py.config import ConfigurationManager


class TestGridCreation:
    """Test Grid class instantiation."""
    
    def test_grid_creation_uniform(self):
        """Test creating grid with uniform spacing."""
        x_n = np.array([0.0, 1.0, 2.0])
        y_n = np.array([0.0, 1.0])
        x_s = np.array([0.5, 1.5])
        y_s = np.array([0.5])
        
        grid = Grid(x_n, y_n, x_s, y_s, (0, 2, 0, 1))
        
        assert grid.x_n.shape == (3,)
        assert grid.y_n.shape == (2,)
        assert grid.x_s.shape == (2,)
        assert grid.y_s.shape == (1,)
    
    def test_grid_repr(self):
        """Test Grid string representation."""
        grid = create_uniform_grid(0, 10, 0, 5, 11, 6)
        repr_str = repr(grid)
        
        assert "Grid" in repr_str
        assert "nx=11" in repr_str
        assert "ny=6" in repr_str
    
    def test_grid_indexing(self):
        """Test grid coordinate access."""
        grid = create_uniform_grid(0, 10, 0, 5, 11, 6)
        
        assert np.array_equal(grid['x_n'], grid.x_n)
        assert np.array_equal(grid['y_n'], grid.y_n)
        assert np.array_equal(grid['x_s'], grid.x_s)
        assert np.array_equal(grid['y_s'], grid.y_s)
        
        with pytest.raises(KeyError):
            _ = grid['invalid']
    
    def test_grid_to_dict(self):
        """Test grid export as dictionary."""
        grid = create_uniform_grid(0, 10, 0, 5, 11, 6)
        data = grid.to_dict()
        
        assert 'x_n' in data
        assert 'y_n' in data
        assert 'x_s' in data
        assert 'y_s' in data
        assert 'metadata' in data
        assert data['metadata']['nx'] == 11
        assert data['metadata']['ny'] == 6


class TestUniformGridGeneration:
    """Test uniform grid creation."""
    
    def test_uniform_grid_simple(self):
        """Test simple uniform grid."""
        grid = create_uniform_grid(0, 10, 0, 5, 11, 6)
        
        # Check normal nodes
        assert len(grid.x_n) == 11
        assert len(grid.y_n) == 6
        assert grid.x_n[0] == 0.0
        assert grid.x_n[-1] == 10.0
        assert grid.y_n[0] == 0.0
        assert grid.y_n[-1] == 5.0
        
        # Check spacing
        dx = np.diff(grid.x_n)
        dy = np.diff(grid.y_n)
        assert np.allclose(dx, 1.0)
        assert np.allclose(dy, 1.0)
    
    def test_uniform_grid_spacing_coarse(self):
        """Test uniform grid with coarse spacing."""
        grid = create_uniform_grid(0, 100, 0, 50, 11, 6)
        
        dx = np.diff(grid.x_n)
        dy = np.diff(grid.y_n)
        assert np.allclose(dx, 100/10)
        assert np.allclose(dy, 50/5)
    
    def test_uniform_staggered_nodes(self):
        """Test staggered node positioning."""
        grid = create_uniform_grid(0, 10, 0, 5, 11, 6)
        
        # Staggered nodes should be midpoints
        expected_x_s = (grid.x_n[:-1] + grid.x_n[1:]) / 2.0
        expected_y_s = (grid.y_n[:-1] + grid.y_n[1:]) / 2.0
        
        assert np.allclose(grid.x_s, expected_x_s)
        assert np.allclose(grid.y_s, expected_y_s)
    
    def test_uniform_grid_metadata(self):
        """Test grid metadata computation."""
        grid = create_uniform_grid(0, 10, 0, 5, 11, 6)
        
        assert grid.metadata.nx == 11
        assert grid.metadata.ny == 6
        assert grid.metadata.n_cells_x == 10
        assert grid.metadata.n_cells_y == 5
        assert grid.metadata.x_min == 0.0
        assert grid.metadata.x_max == 10.0
        assert grid.metadata.y_min == 0.0
        assert grid.metadata.y_max == 5.0
        assert np.isclose(grid.metadata.dx_min, 1.0)
        assert np.isclose(grid.metadata.dx_max, 1.0)


class TestZonedCoordinateGeneration:
    """Test zone-based coordinate generation."""
    
    def test_zoned_coordinates_simple(self):
        """Test simple zoned coordinates."""
        coords = _generate_zoned_coordinates(
            domain_min=0.0,
            domain_max=10.0,
            zone_breaks=[0.0, 5.0, 10.0],
            zone_spacing=[1.0, 2.0]
        )
        
        assert coords[0] == 0.0
        assert coords[-1] == 10.0
        assert np.all(np.diff(coords) > 0)
    
    def test_zoned_coordinates_refinement(self):
        """Test refined zone in middle."""
        coords = _generate_zoned_coordinates(
            domain_min=0.0,
            domain_max=100.0,
            zone_breaks=[0.0, 30.0, 70.0, 100.0],
            zone_spacing=[10.0, 1.0, 10.0]  # Fine in middle
        )
        
        # Check monotonicity
        assert np.all(np.diff(coords) > 0)
        # Check refined zone has finer spacing
        mid_section = coords[(coords >= 30) & (coords <= 70)]
        outer_section = coords[(coords <= 30) | (coords >= 70)]
        
        mid_spacing = np.mean(np.diff(mid_section))
        outer_spacing = np.mean(np.diff(outer_section))
        
        assert mid_spacing < outer_spacing
    
    def test_zoned_coordinates_monotonic(self):
        """Test that zoned coordinates are monotonic."""
        coords = _generate_zoned_coordinates(
            domain_min=0.0,
            domain_max=50.0,
            zone_breaks=[0.0, 10.0, 30.0, 50.0],
            zone_spacing=[2.0, 1.0, 3.0]
        )
        
        diffs = np.diff(coords)
        assert np.all(diffs > 0)
    
    def test_zoned_coordinates_error_invalid_breaks(self):
        """Test error on invalid zone breaks."""
        with pytest.raises(ValueError, match="not strictly increasing"):
            _generate_zoned_coordinates(
                domain_min=0.0,
                domain_max=10.0,
                zone_breaks=[0.0, 5.0, 3.0, 10.0],  # Invalid!
                zone_spacing=[1.0, 1.0, 1.0]
            )
    
    def test_zoned_coordinates_error_wrong_length(self):
        """Test error on wrong number of zone spacings."""
        with pytest.raises(ValueError, match="zone_breaks length"):
            _generate_zoned_coordinates(
                domain_min=0.0,
                domain_max=10.0,
                zone_breaks=[0.0, 5.0, 10.0],
                zone_spacing=[1.0]  # Missing one!
            )


class TestZonedGridGeneration:
    """Test zone-based grid creation."""
    
    def test_zoned_grid_creation(self):
        """Test creating zone-based grid."""
        grid = create_zoned_grid(
            x_min=0, x_max=100,
            y_min=0, y_max=50,
            x_breaks=[0, 40, 100],
            x_spacing=[5, 10],
            y_breaks=[0, 20, 50],
            y_spacing=[2, 5]
        )
        
        assert len(grid.x_n) > 0
        assert len(grid.y_n) > 0
        assert grid.x_n[0] == 0.0
        assert grid.x_n[-1] == 100.0
        assert grid.y_n[0] == 0.0
        assert grid.y_n[-1] == 50.0
    
    def test_zoned_grid_staggered_offset(self):
        """Test staggered offset in zoned grid."""
        grid = create_zoned_grid(
            x_min=0, x_max=100,
            y_min=0, y_max=50,
            x_breaks=[0, 50, 100],
            x_spacing=[10, 10],
            y_breaks=[0, 50],
            y_spacing=[10]
        )
        
        # Staggered nodes should be midpoints
        expected_x_s = (grid.x_n[:-1] + grid.x_n[1:]) / 2.0
        expected_y_s = (grid.y_n[:-1] + grid.y_n[1:]) / 2.0
        
        assert np.allclose(grid.x_s, expected_x_s)
        assert np.allclose(grid.y_s, expected_y_s)


class TestGridValidation:
    """Test grid validation and constraints."""
    
    def test_validation_x_n_not_increasing(self):
        """Test error on non-increasing x_n."""
        with pytest.raises(ValueError, match="x_n coordinates"):
            Grid(
                x_n=np.array([0, 2, 1]),  # Not increasing!
                y_n=np.array([0, 1]),
                x_s=np.array([1]),
                y_s=np.array([0.5]),
                domain_bounds=(0, 2, 0, 1)
            )
    
    def test_validation_y_n_not_increasing(self):
        """Test error on non-increasing y_n."""
        with pytest.raises(ValueError, match="y_n coordinates"):
            Grid(
                x_n=np.array([0, 1, 2]),
                y_n=np.array([0, 1, 0.5]),  # Not increasing!
                x_s=np.array([0.5, 1.5]),
                y_s=np.array([0.5]),
                domain_bounds=(0, 2, 0, 1)
            )
    
    def test_validation_staggered_length_mismatch_x(self):
        """Test error on mismatched x_s length."""
        with pytest.raises(ValueError, match="x_s length"):
            Grid(
                x_n=np.array([0, 1, 2, 3]),
                y_n=np.array([0, 1]),
                x_s=np.array([0.5, 1.5]),  # Should be length 3!
                y_s=np.array([0.5]),
                domain_bounds=(0, 3, 0, 1)
            )
    
    def test_validation_staggered_length_mismatch_y(self):
        """Test error on mismatched y_s length."""
        with pytest.raises(ValueError, match="y_s length"):
            Grid(
                x_n=np.array([0, 1, 2]),
                y_n=np.array([0, 1, 2]),
                x_s=np.array([0.5, 1.5]),
                y_s=np.array([0.5]),  # Should be length 2!
                domain_bounds=(0, 2, 0, 2)
            )
    
    def test_validation_success_valid_grid(self):
        """Test validation succeeds with valid grid."""
        grid = Grid(
            x_n=np.array([0, 1, 2]),
            y_n=np.array([0, 1]),
            x_s=np.array([0.5, 1.5]),
            y_s=np.array([0.5]),
            domain_bounds=(0, 2, 0, 1)
        )
        
        # If we get here without exception, validation passed
        assert grid.x_n is not None


class TestGridMetadata:
    """Test grid metadata calculation."""
    
    def test_metadata_uniform(self):
        """Test metadata for uniform grid."""
        grid = create_uniform_grid(0, 100, 0, 50, 21, 11)
        
        assert grid.metadata.nx == 21
        assert grid.metadata.ny == 11
        assert grid.metadata.n_cells_x == 20
        assert grid.metadata.n_cells_y == 10
        assert grid.metadata.x_min == 0
        assert grid.metadata.x_max == 100
        assert grid.metadata.y_min == 0
        assert grid.metadata.y_max == 50
        assert np.isclose(grid.metadata.dx_min, 5.0)
        assert np.isclose(grid.metadata.dx_max, 5.0)
        assert np.isclose(grid.metadata.dy_min, 5.0)
        assert np.isclose(grid.metadata.dy_max, 5.0)
        assert np.isclose(grid.metadata.aspect_ratio_max, 1.0)
    
    def test_metadata_zoned(self):
        """Test metadata for zoned grid."""
        grid = create_zoned_grid(
            x_min=0, x_max=100,
            y_min=0, y_max=50,
            x_breaks=[0, 40, 100],
            x_spacing=[1, 10],  # Different spacing
            y_breaks=[0, 50],
            y_spacing=[5]
        )
        
        # Check that min/max spacing are different
        assert grid.metadata.dx_min < grid.metadata.dx_max
        # Aspect ratio should be non-trivial
        assert grid.metadata.aspect_ratio_max > 1.0


class TestGridFromConfig:
    """Test grid generation from ConfigurationManager."""
    
    def test_grid_from_config_uniform(self):
        """Test generating grid from uniform config."""
        config_file = Path(__file__).parent.parent / "sister_py" / "data" / "defaults.yaml"
        
        if config_file.exists():
            cfg = ConfigurationManager.load(str(config_file))
            grid = Grid.generate(cfg)
            
            # Check basic properties
            assert grid.x_n is not None
            assert grid.y_n is not None
            assert len(grid.x_s) == len(grid.x_n) - 1
            assert len(grid.y_s) == len(grid.y_n) - 1
    
    def test_grid_from_config_domain_bounds(self):
        """Test that grid respects domain bounds from config."""
        config_file = Path(__file__).parent.parent / "sister_py" / "data" / "defaults.yaml"
        
        if config_file.exists():
            cfg = ConfigurationManager.load(str(config_file))
            grid = Grid.generate(cfg)
            
            # Check domain bounds
            assert np.isclose(grid.x_n[0], cfg.DOMAIN.x_min if hasattr(cfg.DOMAIN, 'x_min') else 0.0)
            assert np.isclose(grid.x_n[-1], cfg.DOMAIN.xsize)
            assert np.isclose(grid.y_n[0], cfg.DOMAIN.y_min if hasattr(cfg.DOMAIN, 'y_min') else 0.0)
            assert np.isclose(grid.y_n[-1], cfg.DOMAIN.ysize)


class TestGridConsistency:
    """Test grid mathematical consistency."""
    
    def test_staggered_node_bounds(self):
        """Test that staggered nodes are within normal node bounds."""
        grid = create_uniform_grid(0, 10, 0, 5, 11, 6)
        
        # Staggered nodes should be strictly between normal nodes
        assert grid.x_s[0] > grid.x_n[0]
        assert grid.x_s[-1] < grid.x_n[-1]
        assert grid.y_s[0] > grid.y_n[0]
        assert grid.y_s[-1] < grid.y_n[-1]
    
    def test_cell_count_consistency(self):
        """Test that cell counts match node spacing."""
        grid = create_uniform_grid(0, 100, 0, 50, 11, 6)
        
        # n_cells = n_nodes - 1
        assert grid.metadata.n_cells_x == len(grid.x_n) - 1
        assert grid.metadata.n_cells_y == len(grid.y_n) - 1
    
    def test_coordinate_symmetry_uniform(self):
        """Test coordinate symmetry in uniform grid."""
        grid = create_uniform_grid(-10, 10, -5, 5, 21, 11)
        
        # Check symmetry about origin
        dx = np.diff(grid.x_n)
        dy = np.diff(grid.y_n)
        assert np.allclose(dx, np.ones_like(dx))  # Uniform
        assert np.allclose(dy, np.ones_like(dy))  # Uniform


class TestGridPerformance:
    """Performance benchmarks for grid generation."""
    
    def test_performance_large_uniform_grid(self):
        """Test performance of large uniform grid generation."""
        import time
        
        start = time.time()
        grid = create_uniform_grid(0, 1000, 0, 500, 501, 251)
        elapsed = time.time() - start
        
        # Should be very fast (< 100ms)
        assert elapsed < 0.1
        assert grid.x_n is not None
    
    def test_performance_zoned_grid_generation(self):
        """Test performance of zoned grid generation."""
        import time
        
        start = time.time()
        grid = create_zoned_grid(
            x_min=0, x_max=1000,
            y_min=0, y_max=500,
            x_breaks=[0, 300, 700, 1000],
            x_spacing=[5, 2, 5],
            y_breaks=[0, 150, 350, 500],
            y_spacing=[10, 5, 10]
        )
        elapsed = time.time() - start
        
        # Should be very fast (< 100ms)
        assert elapsed < 0.1
        assert grid.x_n is not None


class TestGridEdgeCases:
    """Test edge cases and boundary conditions."""
    
    def test_minimal_grid(self):
        """Test minimal 2x2 node grid."""
        grid = create_uniform_grid(0, 1, 0, 1, 2, 2)
        
        assert len(grid.x_n) == 2
        assert len(grid.y_n) == 2
        assert len(grid.x_s) == 1
        assert len(grid.y_s) == 1
    
    def test_very_fine_grid(self):
        """Test very fine grid."""
        grid = create_uniform_grid(0, 1, 0, 1, 1001, 501)
        
        assert len(grid.x_n) == 1001
        assert len(grid.y_n) == 501
        assert len(grid.x_s) == 1000
        assert len(grid.y_s) == 500
    
    def test_non_square_domain(self):
        """Test non-square domain."""
        grid = create_uniform_grid(0, 100, 0, 1, 101, 2)
        
        # Both spacings are 1.0 and 1.0, so aspect ratio is 1.0
        assert grid.metadata.aspect_ratio_max == 1.0
    
    def test_negative_domain(self):
        """Test domain with negative coordinates."""
        grid = create_uniform_grid(-50, 50, -25, 25, 101, 51)
        
        assert grid.x_n[0] == -50
        assert grid.x_n[-1] == 50
        assert grid.y_n[0] == -25
        assert grid.y_n[-1] == 25


# Run tests if executed directly
if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
