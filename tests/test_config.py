"""
Comprehensive test suite for SiSteR-py configuration system.

Tests cover:
- Pydantic model validation
- ConfigurationManager loading and export
- Material property calculations
- Round-trip YAML operations
- Performance characteristics
- Edge cases and error handling
"""

import pytest
import os
import tempfile
import time
from pathlib import Path

from pydantic import ValidationError
from sister_py.config import (
    ConfigurationManager,
    Material,
    SimulationConfig,
    DomainConfig,
    GridConfig,
    DensityParams,
    DuctileCreepParams,
    RheologyConfig,
    PlasticityParams,
    ElasticityParams,
    ThermalParams,
    MaterialConfig,
    BCConfig,
    PhysicsConfig,
    SolverConfig,
    FullConfig,
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def continental_rift_path():
    """Path to continental rift example."""
    return Path(__file__).parent.parent / "sister_py" / "data" / "examples" / "continental_rift.yaml"


@pytest.fixture
def temp_yaml():
    """Create temporary YAML file."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yield f.name
    try:
        os.unlink(f.name)
    except:
        pass


# ============================================================================
# Unit Tests: Pydantic Models
# ============================================================================

class TestSimulationConfig:
    """Test SimulationConfig validation."""
    
    def test_valid_simulation(self):
        """Valid simulation config should load."""
        cfg = SimulationConfig(Nt=1000, dt_out=50, output_dir="./results")
        assert cfg.Nt == 1000
        assert cfg.dt_out == 50
    
    def test_invalid_nt_zero(self):
        """Nt must be > 0."""
        with pytest.raises(ValidationError) as exc_info:
            SimulationConfig(Nt=0, dt_out=50, output_dir="./results")
        assert "greater than 0" in str(exc_info.value)
    
    def test_invalid_nt_negative(self):
        """Nt must be positive."""
        with pytest.raises(ValidationError):
            SimulationConfig(Nt=-100, dt_out=50, output_dir="./results")


class TestDomainConfig:
    """Test DomainConfig validation."""
    
    def test_valid_domain(self):
        """Valid domain should load."""
        cfg = DomainConfig(xsize=100e3, ysize=50e3)
        assert cfg.xsize == 100e3
        assert cfg.ysize == 50e3
    
    def test_invalid_xsize_zero(self):
        """xsize must be > 0."""
        with pytest.raises(ValidationError):
            DomainConfig(xsize=0, ysize=50e3)
    
    def test_invalid_ysize_negative(self):
        """ysize must be positive."""
        with pytest.raises(ValidationError):
            DomainConfig(xsize=100e3, ysize=-50e3)


class TestGridConfig:
    """Test GridConfig validation."""
    
    def test_valid_grid(self):
        """Valid grid config should load."""
        cfg = GridConfig(
            x_spacing=[1000, 500, 1000],
            x_breaks=[50e3, 150e3],
            y_spacing=[1000, 500, 1000],
            y_breaks=[30e3, 70e3]
        )
        assert len(cfg.x_spacing) == 3
        assert len(cfg.x_breaks) == 2
    
    def test_invalid_spacing_zero(self):
        """Grid spacing must be positive."""
        with pytest.raises(ValidationError) as exc_info:
            GridConfig(
                x_spacing=[1000, 0, 1000],
                x_breaks=[50e3, 150e3],
                y_spacing=[1000, 500, 1000],
                y_breaks=[30e3, 70e3]
            )
        assert "positive" in str(exc_info.value).lower()
    
    def test_invalid_spacing_negative(self):
        """Grid spacing must be > 0."""
        with pytest.raises(ValidationError):
            GridConfig(
                x_spacing=[1000, -500, 1000],
                x_breaks=[50e3, 150e3],
                y_spacing=[1000, 500, 1000],
                y_breaks=[30e3, 70e3]
            )
    
    def test_invalid_breaks_not_increasing(self):
        """Boundaries must be strictly increasing."""
        with pytest.raises(ValidationError) as exc_info:
            GridConfig(
                x_spacing=[1000, 500, 1000],
                x_breaks=[150e3, 50e3],  # Reversed!
                y_spacing=[1000, 500, 1000],
                y_breaks=[30e3, 70e3]
            )
        assert "increasing" in str(exc_info.value).lower()
    
    def test_invalid_breaks_equal(self):
        """Boundaries must be strictly increasing (not equal)."""
        with pytest.raises(ValidationError):
            GridConfig(
                x_spacing=[1000, 500, 1000],
                x_breaks=[50e3, 50e3],  # Equal!
                y_spacing=[1000, 500, 1000],
                y_breaks=[30e3, 70e3]
            )


class TestDuctileCreepParams:
    """Test DuctileCreepParams validation."""
    
    def test_valid_creep(self):
        """Valid creep params should load."""
        cfg = DuctileCreepParams(A=1e-21, E=400e3, n=3.5)
        assert cfg.A == 1e-21
        assert cfg.E == 400e3
        assert cfg.n == 3.5
    
    def test_invalid_a_zero(self):
        """A must be > 0."""
        with pytest.raises(ValidationError):
            DuctileCreepParams(A=0, E=400e3, n=3.5)
    
    def test_invalid_n_zero(self):
        """n must be > 0."""
        with pytest.raises(ValidationError):
            DuctileCreepParams(A=1e-21, E=400e3, n=0)
    
    def test_valid_e_zero(self):
        """E can be 0 (diffusion creep)."""
        cfg = DuctileCreepParams(A=0.5e-18, E=0, n=1.0)
        assert cfg.E == 0


class TestPlasticityParams:
    """Test PlasticityParams validation."""
    
    def test_valid_plasticity(self):
        """Valid plasticity params should load."""
        cfg = PlasticityParams(C=40e6, mu=0.6)
        assert cfg.C == 40e6
        assert cfg.mu == 0.6
    
    def test_invalid_mu_zero(self):
        """mu must be > 0."""
        with pytest.raises(ValidationError):
            PlasticityParams(C=40e6, mu=0)
    
    def test_invalid_mu_one(self):
        """mu must be < 1."""
        with pytest.raises(ValidationError):
            PlasticityParams(C=40e6, mu=1.0)
    
    def test_invalid_mu_greater_than_one(self):
        """mu must be < 1."""
        with pytest.raises(ValidationError) as exc_info:
            PlasticityParams(C=40e6, mu=1.5)
        assert "less than 1" in str(exc_info.value)
    
    def test_valid_c_zero(self):
        """C can be 0 (no cohesion)."""
        cfg = PlasticityParams(C=0, mu=0.6)
        assert cfg.C == 0


class TestMaterialConfig:
    """Test MaterialConfig validation."""
    
    def test_valid_material(self):
        """Valid material config should load."""
        cfg = MaterialConfig(
            phase=1,
            name="Test Material",
            density=DensityParams(rho0=3000, alpha=3e-5),
            plasticity=PlasticityParams(C=0, mu=0.6)
        )
        assert cfg.phase == 1
        assert cfg.name == "Test Material"
    
    def test_invalid_phase_zero(self):
        """Phase must be > 0."""
        with pytest.raises(ValidationError):
            MaterialConfig(
                phase=0,
                name="Test",
                density=DensityParams(rho0=3000, alpha=0)
            )


class TestFullConfig:
    """Test FullConfig cross-model validation."""
    
    def test_invalid_duplicate_phases(self):
        """Phase IDs must be unique."""
        with pytest.raises(ValidationError) as exc_info:
            FullConfig(
                SIMULATION=SimulationConfig(Nt=100, dt_out=10, output_dir="./results"),
                DOMAIN=DomainConfig(xsize=100e3, ysize=100e3),
                GRID=GridConfig(
                    x_spacing=[5000], x_breaks=[0, 100e3],
                    y_spacing=[5000], y_breaks=[0, 100e3]
                ),
                MATERIALS=[
                    MaterialConfig(
                        phase=1,
                        name="Material1",
                        density=DensityParams(rho0=3000, alpha=0)
                    ),
                    MaterialConfig(
                        phase=1,  # Duplicate!
                        name="Material2",
                        density=DensityParams(rho0=3000, alpha=0)
                    )
                ],
                BC={},
                PHYSICS=PhysicsConfig(elasticity=True, plasticity=True, thermal=False),
                SOLVER=SolverConfig(Npicard_min=5, Npicard_max=50, conv_tol=1e-6, switch_to_newton=0)
            )
        assert "unique" in str(exc_info.value).lower()


# ============================================================================
# Unit Tests: Material Class
# ============================================================================

class TestMaterial:
    """Test Material property calculations."""
    
    @pytest.fixture
    def mantle_material(self):
        """Create a mantle material for testing."""
        cfg = MaterialConfig(
            phase=2,
            name="Mantle",
            density=DensityParams(rho0=3300, alpha=3e-5),
            rheology=RheologyConfig(
                type="ductile",
                diffusion=DuctileCreepParams(A=1e-21, E=400e3, n=3.5),
                dislocation=DuctileCreepParams(A=1.9e-16, E=540e3, n=3.5)
            ),
            plasticity=PlasticityParams(C=0, mu=0.6),
            elasticity=ElasticityParams(G=6.4e10),
            thermal=ThermalParams(k=4.5, cp=1250)
        )
        return Material(cfg)
    
    def test_material_properties(self, mantle_material):
        """Material should expose phase and name."""
        assert mantle_material.phase == 2
        assert mantle_material.name == "Mantle"
    
    def test_density(self, mantle_material):
        """Density should decrease with temperature."""
        rho_cold = mantle_material.density(273)  # 273 K
        rho_hot = mantle_material.density(1273)  # 1273 K
        assert rho_cold > rho_hot
        assert rho_cold == pytest.approx(3300 * (1 - 3e-5 * 273), rel=1e-6)
    
    def test_viscosity_ductile_positive(self, mantle_material):
        """Ductile viscosity should be positive."""
        eta = mantle_material.viscosity_ductile(
            sigma_II=1e7,  # 10 MPa
            eps_II=1e-15,  # Very small strain rate
            T=1373  # 1100 °C
        )
        assert eta > 0
        assert eta != float('inf')
    
    def test_viscosity_ductile_temperature_dependence(self, mantle_material):
        """Warmer material should be weaker (lower viscosity)."""
        eta_cold = mantle_material.viscosity_ductile(sigma_II=1e7, eps_II=1e-15, T=273)
        eta_hot = mantle_material.viscosity_ductile(sigma_II=1e7, eps_II=1e-15, T=1373)
        assert eta_cold > eta_hot
    
    def test_viscosity_plastic_below_yield(self, mantle_material):
        """Below yield, plastic viscosity should be infinite."""
        eta = mantle_material.viscosity_plastic(sigma_II=1e6, P=1e9)
        assert eta == float('inf')
    
    def test_viscosity_plastic_above_yield(self, mantle_material):
        """Above yield, plastic viscosity should be finite."""
        # Yield: sigma_Y = (0 + 0.6 * 1e9) * cos(arctan(0.6))
        eta = mantle_material.viscosity_plastic(sigma_II=1e9, P=1e9)
        assert eta > 0
        assert eta != float('inf')
    
    def test_viscosity_effective(self, mantle_material):
        """Effective viscosity should be minimum of ductile and plastic."""
        eta_eff = mantle_material.viscosity_effective(
            sigma_II=1e8,
            eps_II=1e-15,
            T=1373,
            P=1e9
        )
        assert eta_eff > 0


# ============================================================================
# Integration Tests: ConfigurationManager
# ============================================================================

class TestConfigurationManager:
    """Test ConfigurationManager loading and export."""
    
    def test_load_continental_rift(self, continental_rift_path):
        """Load continental rift example."""
        if not continental_rift_path.exists():
            pytest.skip(f"Example file not found: {continental_rift_path}")
        
        cfg = ConfigurationManager.load(str(continental_rift_path))
        assert cfg.DOMAIN.xsize == 170e3
        assert cfg.DOMAIN.ysize == 60e3
        assert len(cfg.MATERIALS) == 2
    
    def test_nested_attribute_access(self, continental_rift_path):
        """Nested attribute access should work."""
        if not continental_rift_path.exists():
            pytest.skip(f"Example file not found: {continental_rift_path}")
        
        cfg = ConfigurationManager.load(str(continental_rift_path))
        assert cfg.DOMAIN.xsize == 170e3
        assert cfg.SIMULATION.Nt == 1600
    
    def test_get_materials(self, continental_rift_path):
        """get_materials() should return dict of Material objects."""
        if not continental_rift_path.exists():
            pytest.skip(f"Example file not found: {continental_rift_path}")
        
        cfg = ConfigurationManager.load(str(continental_rift_path))
        materials = cfg.get_materials()
        
        assert isinstance(materials, dict)
        assert 1 in materials
        assert 2 in materials
        assert isinstance(materials[1], Material)
        assert materials[1].name == "Sticky Layer"
    
    def test_to_dict(self, continental_rift_path):
        """to_dict() should return JSON-serializable dict."""
        if not continental_rift_path.exists():
            pytest.skip(f"Example file not found: {continental_rift_path}")
        
        cfg = ConfigurationManager.load(str(continental_rift_path))
        data = cfg.to_dict()
        
        assert isinstance(data, dict)
        assert 'DOMAIN' in data
        assert 'MATERIALS' in data
        assert data['DOMAIN']['xsize'] == 170e3
    
    def test_to_string(self, continental_rift_path):
        """to_string() should return YAML-formatted string."""
        if not continental_rift_path.exists():
            pytest.skip(f"Example file not found: {continental_rift_path}")
        
        cfg = ConfigurationManager.load(str(continental_rift_path))
        yaml_str = cfg.to_string()
        
        assert isinstance(yaml_str, str)
        assert 'DOMAIN' in yaml_str
        assert 'xsize' in yaml_str
    
    def test_round_trip(self, continental_rift_path, temp_yaml):
        """Load → dict → save → load should preserve values."""
        if not continental_rift_path.exists():
            pytest.skip(f"Example file not found: {continental_rift_path}")
        
        # Load original
        cfg1 = ConfigurationManager.load(str(continental_rift_path))
        xsize1 = cfg1.DOMAIN.xsize
        
        # Export and reload
        cfg1.to_yaml(temp_yaml)
        cfg2 = ConfigurationManager.load(temp_yaml)
        xsize2 = cfg2.DOMAIN.xsize
        
        assert xsize1 == xsize2
    
    def test_validate(self, continental_rift_path):
        """validate() should re-validate config."""
        if not continental_rift_path.exists():
            pytest.skip(f"Example file not found: {continental_rift_path}")
        
        cfg = ConfigurationManager.load(str(continental_rift_path))
        # Should not raise
        cfg.validate()


# ============================================================================
# Performance Tests
# ============================================================================

class TestPerformance:
    """Test performance characteristics."""
    
    def test_load_performance(self, continental_rift_path):
        """Config load should be < 100 ms."""
        if not continental_rift_path.exists():
            pytest.skip(f"Example file not found: {continental_rift_path}")
        
        t_start = time.time()
        cfg = ConfigurationManager.load(str(continental_rift_path))
        t_elapsed = (time.time() - t_start) * 1000  # Convert to ms
        
        # Should load in < 100 ms
        assert t_elapsed < 100, f"Load took {t_elapsed:.1f} ms (expected < 100 ms)"
    
    def test_viscosity_performance(self, continental_rift_path):
        """Single viscosity call should be < 10 µs."""
        if not continental_rift_path.exists():
            pytest.skip(f"Example file not found: {continental_rift_path}")
        
        cfg = ConfigurationManager.load(str(continental_rift_path))
        materials = cfg.get_materials()
        mantle = materials[2]
        
        # Time 1000 calls
        t_start = time.time()
        for _ in range(1000):
            _ = mantle.viscosity_ductile(sigma_II=1e7, eps_II=1e-15, T=1373)
        t_elapsed = (time.time() - t_start) * 1e6 / 1000  # Convert to µs per call
        
        # Should be < 10 µs per call
        assert t_elapsed < 10, f"Viscosity calc took {t_elapsed:.2f} µs (expected < 10 µs)"


# ============================================================================
# Error Handling Tests
# ============================================================================

class TestErrorHandling:
    """Test error messages and edge cases."""
    
    def test_file_not_found(self):
        """Loading nonexistent file should raise FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            ConfigurationManager.load("/nonexistent/path/config.yaml")
    
    def test_invalid_yaml_syntax(self, temp_yaml):
        """Invalid YAML should raise error."""
        with open(temp_yaml, 'w') as f:
            f.write("INVALID: YAML: [SYNTAX:")
        
        with pytest.raises(Exception):  # yaml.YAMLError or similar
            ConfigurationManager.load(temp_yaml)
    
    def test_missing_required_field(self, temp_yaml):
        """Missing required field should raise ValidationError."""
        with open(temp_yaml, 'w') as f:
            f.write("SIMULATION:\n  Nt: 100\n")  # Missing dt_out, output_dir
        
        with pytest.raises(ValidationError):
            ConfigurationManager.load(temp_yaml)
    
    def test_invalid_type(self, temp_yaml):
        """Wrong type should raise ValidationError."""
        with open(temp_yaml, 'w') as f:
            f.write("""
SIMULATION:
  Nt: "not_an_integer"
  dt_out: 50
  output_dir: "./results"
DOMAIN:
  xsize: 100e3
  ysize: 100e3
GRID:
  x_spacing: [1000]
  x_breaks: [100e3]
  y_spacing: [1000]
  y_breaks: [100e3]
MATERIALS: []
BC: {}
PHYSICS:
  elasticity: true
  plasticity: true
  thermal: false
SOLVER:
  Npicard_min: 5
  Npicard_max: 50
  conv_tol: 1e-6
  switch_to_newton: 0
""")
        
        with pytest.raises(ValidationError):
            ConfigurationManager.load(temp_yaml)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
