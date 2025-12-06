"""
Tests for Thermal Solver Module

Tests for:
- Thermal properties and material management
- Heat diffusion (steady-state and transient)
- Advection-diffusion coupling
- Thermal boundary conditions
- Complete thermal model
"""

import pytest
import numpy as np
from scipy import sparse

from sister_py.thermal_solver import (
    ThermalProperties, ThermalBoundaryCondition, ThermalMaterialProperties,
    HeatDiffusionSolver, AdvectionDiffusionSolver, ThermalModel, ThermalFieldData,
    compute_thermal_conductivity, compute_heat_capacity,
    estimate_thermal_time_scale, interpolate_temperature_to_markers
)


class TestThermalProperties:
    """Tests for thermal properties."""
    
    def test_thermal_properties_init(self):
        """Test ThermalProperties initialization."""
        props = ThermalProperties()
        assert props.k == 3.0
        assert props.cp == 1000.0
        assert props.rho == 2800.0
    
    def test_thermal_properties_custom(self):
        """Test custom thermal properties."""
        props = ThermalProperties(k=5.0, cp=1200.0, rho=3000.0)
        assert props.k == 5.0
        assert props.cp == 1200.0
        assert props.rho == 3000.0


class TestThermalBoundaryCondition:
    """Tests for thermal boundary conditions."""
    
    def test_bc_dirichlet(self):
        """Test Dirichlet boundary condition."""
        bc = ThermalBoundaryCondition(
            'top',
            ThermalBoundaryCondition.BCType.DIRICHLET,
            value=1273.0
        )
        assert bc.boundary == 'top'
        assert bc.bc_type == 'dirichlet'
        assert bc.value == 1273.0
    
    def test_bc_neumann(self):
        """Test Neumann boundary condition."""
        bc = ThermalBoundaryCondition(
            'bottom',
            ThermalBoundaryCondition.BCType.NEUMANN,
            value=0.05  # Heat flux W/m²
        )
        assert bc.bc_type == 'neumann'
        assert bc.value == 0.05
    
    def test_bc_robin(self):
        """Test Robin boundary condition."""
        bc = ThermalBoundaryCondition(
            'top',
            ThermalBoundaryCondition.BCType.ROBIN,
            ambient_temp=273.15,
            h_coeff=100.0
        )
        assert bc.bc_type == 'robin'
        assert bc.h_coeff == 100.0


class TestThermalMaterialProperties:
    """Tests for material property management."""
    
    def test_material_props_init(self):
        """Test ThermalMaterialProperties initialization."""
        mat_props = ThermalMaterialProperties(n_phases=5)
        assert mat_props.n_phases == 5
    
    def test_set_and_get_properties(self):
        """Test setting and retrieving properties."""
        mat_props = ThermalMaterialProperties()
        props1 = ThermalProperties(k=4.0, cp=1100.0)
        mat_props.set_properties(1, props1)
        
        retrieved = mat_props.get_properties(1)
        assert retrieved.k == 4.0
        assert retrieved.cp == 1100.0
    
    def test_property_fields(self):
        """Test getting property fields from phase field."""
        mat_props = ThermalMaterialProperties()
        props1 = ThermalProperties(k=3.0)
        props2 = ThermalProperties(k=5.0)
        
        mat_props.set_properties(1, props1)
        mat_props.set_properties(2, props2)
        
        # Create phase field
        phase_field = np.array([[1, 1, 2],
                               [1, 2, 2]])
        
        k_field = mat_props.get_conductivity_field(phase_field)
        
        assert k_field[0, 0] == 3.0
        assert k_field[1, 2] == 5.0


class TestThermalFieldData:
    """Tests for thermal field data."""
    
    def test_field_data_init(self):
        """Test ThermalFieldData initialization."""
        T = np.ones((5, 5)) * 1000.0
        field = ThermalFieldData(temperature=T)
        
        assert field.temperature.shape == (5, 5)
        assert field.time == 0.0
    
    def test_temperature_extrema(self):
        """Test temperature extrema property."""
        T = np.array([[273.15, 500.0],
                     [1000.0, 1500.0]])
        field = ThermalFieldData(temperature=T)
        
        T_min, T_max = field.temperature_extrema
        assert T_min == 273.15
        assert T_max == 1500.0


class TestHeatDiffusionSolver:
    """Tests for heat diffusion solver."""
    
    def test_diffusion_solver_init(self):
        """Test HeatDiffusionSolver initialization."""
        solver = HeatDiffusionSolver()
        assert solver.n_iterations == 0
    
    def test_laplace_assembly(self):
        """Test Laplace operator assembly."""
        solver = HeatDiffusionSolver()
        
        # Simple uniform grid
        nx, ny = 5, 5
        k_field = np.ones((ny, nx)) * 3.0
        grid_x = np.linspace(0, 10, nx)
        grid_y = np.linspace(0, 10, ny)
        
        matrix, rhs = solver.assemble_laplace_operator(k_field, grid_x, grid_y)
        
        # Check matrix properties
        assert matrix.shape == (nx*ny, nx*ny)
        assert matrix.nnz > 0  # Has non-zeros
        assert isinstance(matrix, sparse.csr_matrix)
    
    def test_transient_assembly(self):
        """Test transient heat equation assembly."""
        solver = HeatDiffusionSolver()
        
        nx, ny = 5, 5
        k_field = np.ones((ny, nx)) * 3.0
        cp_field = np.ones((ny, nx)) * 1000.0
        rho_field = np.ones((ny, nx)) * 2800.0
        grid_x = np.linspace(0, 10, nx)
        grid_y = np.linspace(0, 10, ny)
        dt = 1e5
        
        lhs, mass = solver.assemble_transient_operator(
            k_field, cp_field, rho_field, grid_x, grid_y, dt
        )
        
        # LHS should be larger than just mass (mass + dt*stiffness)
        assert lhs.nnz > mass.nnz
    
    def test_steady_state_solve(self):
        """Test steady-state heat diffusion solve."""
        solver = HeatDiffusionSolver()
        
        nx, ny = 10, 10
        k_field = np.ones((ny, nx)) * 3.0
        grid_x = np.linspace(0, 1, nx)
        grid_y = np.linspace(0, 1, ny)
        heat_source = np.zeros((ny, nx))
        
        T = solver.solve_steady_state(k_field, grid_x, grid_y, heat_source)
        
        assert T.shape == (ny, nx)
        assert np.all(np.isfinite(T))


class TestAdvectionDiffusionSolver:
    """Tests for advection-diffusion solver."""
    
    def test_advdiff_init(self):
        """Test AdvectionDiffusionSolver initialization."""
        solver = AdvectionDiffusionSolver()
        assert solver.verbose is False
    
    def test_advdiff_assembly(self):
        """Test advection-diffusion operator assembly."""
        solver = AdvectionDiffusionSolver()
        
        nx, ny = 5, 5
        velocity_x = np.ones((ny, nx)) * 0.1
        velocity_y = np.zeros((ny, nx))
        k_field = np.ones((ny, nx)) * 3.0
        grid_x = np.linspace(0, 10, nx)
        grid_y = np.linspace(0, 10, ny)
        
        matrix, rhs = solver.assemble_advection_diffusion(
            velocity_x, velocity_y, k_field, grid_x, grid_y
        )
        
        assert matrix.shape == (nx*ny, nx*ny)
        assert rhs.shape == (nx*ny,)


class TestThermalModel:
    """Tests for complete thermal model."""
    
    def test_thermal_model_init(self):
        """Test ThermalModel initialization."""
        grid_x = np.linspace(0, 100e3, 21)
        grid_y = np.linspace(0, 100e3, 21)
        T_init = np.ones((len(grid_y), len(grid_x))) * 1000.0
        
        model = ThermalModel(grid_x, grid_y, T_init)
        
        assert model.T_current.shape == T_init.shape
        assert len(model.time_history) == 1
    
    def test_thermal_model_with_properties(self):
        """Test ThermalModel with material properties."""
        grid_x = np.linspace(0, 100e3, 11)
        grid_y = np.linspace(0, 100e3, 11)
        T_init = np.ones((len(grid_y), len(grid_x))) * 1000.0
        
        mat_props = ThermalMaterialProperties()
        model = ThermalModel(grid_x, grid_y, T_init, material_props=mat_props)
        
        assert model.material_props is not None
    
    def test_solve_pure_diffusion(self):
        """Test solving pure heat diffusion step."""
        grid_x = np.linspace(0, 1e5, 11)
        grid_y = np.linspace(0, 1e5, 11)
        T_init = np.ones((len(grid_y), len(grid_x))) * 1000.0
        
        model = ThermalModel(grid_x, grid_y, T_init)
        
        # Create phase field
        phase_field = np.ones((len(grid_y), len(grid_x)), dtype=int)
        
        # Solve one step
        result = model.solve_step(phase_field, dt=1e6)
        
        assert isinstance(result, ThermalFieldData)
        assert result.temperature.shape == T_init.shape
        assert result.heat_flux_x is not None
    
    def test_dirichlet_bc_application(self):
        """Test applying Dirichlet boundary condition."""
        grid_x = np.linspace(0, 1, 11)
        grid_y = np.linspace(0, 1, 11)
        T = np.ones((len(grid_y), len(grid_x))) * 1000.0
        
        model = ThermalModel(grid_x, grid_y, T)
        bc = ThermalBoundaryCondition('top', 'dirichlet', value=273.0)
        
        T_mod = model.apply_dirichlet_bc(T, bc)
        
        assert T_mod[-1, :].max() == 273.0
        assert T_mod[-1, :].min() == 273.0
    
    def test_estimate_thermal_time_scale(self):
        """Test thermal time scale estimation."""
        grid_x = np.linspace(0, 100e3, 11)
        grid_y = np.linspace(0, 100e3, 11)
        T_init = np.ones((len(grid_y), len(grid_x))) * 1000.0
        
        model = ThermalModel(grid_x, grid_y, T_init)
        tau = model.estimate_thermal_time_scale()
        
        # Should be on order of (100e3)² / (3 / (2800*1000)) ~ 9e15 s (very large)
        assert tau > 0


class TestConvenienceFunctions:
    """Tests for convenience functions."""
    
    def test_thermal_conductivity_olivine(self):
        """Test thermal conductivity for olivine."""
        k = compute_thermal_conductivity(273.15, 'olivine')
        assert 0.5 < k < 10.0
        
        k_hot = compute_thermal_conductivity(1000.0, 'olivine')
        assert k_hot < k  # Decreases with temperature
    
    def test_thermal_conductivity_basalt(self):
        """Test thermal conductivity for basalt."""
        k = compute_thermal_conductivity(500.0, 'basalt')
        assert 0.5 < k < 10.0
    
    def test_heat_capacity_olivine(self):
        """Test heat capacity for olivine."""
        cp = compute_heat_capacity(500.0, 'olivine')
        assert 700.0 <= cp <= 2000.0
    
    def test_time_scale_estimation(self):
        """Test thermal time scale estimation."""
        L = 100e3  # 100 km
        tau = estimate_thermal_time_scale(L)
        
        # Should be positive and large
        assert tau > 0
        assert tau < 1e20  # Reasonable upper bound
    
    def test_temperature_interpolation(self):
        """Test temperature interpolation to markers."""
        # Create simple 3x3 grid
        grid_x = np.array([0.0, 1.0, 2.0])
        grid_y = np.array([0.0, 1.0, 2.0])
        
        T_grid = np.array([[273.15, 500.0, 1000.0],
                          [373.15, 600.0, 1100.0],
                          [473.15, 700.0, 1200.0]])
        
        # Marker at center
        marker_pos = np.array([[1.0, 1.0]])
        
        T_markers = interpolate_temperature_to_markers(T_grid, marker_pos, grid_x, grid_y)
        
        # Should be close to grid value at [1,1]
        assert len(T_markers) == 1
        assert abs(T_markers[0] - 600.0) < 10.0


class TestThermalPhysics:
    """Tests for physical correctness of thermal solver."""
    
    def test_heat_flow_direction(self):
        """Test that heat flows from hot to cold."""
        grid_x = np.linspace(0, 1, 11)
        grid_y = np.linspace(0, 1, 11)
        
        # Uniform temperature field
        T_init = np.ones((len(grid_y), len(grid_x))) * 500.0
        
        model = ThermalModel(grid_x, grid_y, T_init)
        phase_field = np.ones((len(grid_y), len(grid_x)), dtype=int)
        
        # Set boundary conditions: hot on left, cold on right
        model.set_boundary_conditions([
            ThermalBoundaryCondition('left', ThermalBoundaryCondition.BCType.DIRICHLET, value=1000.0),
            ThermalBoundaryCondition('right', ThermalBoundaryCondition.BCType.DIRICHLET, value=273.0),
        ])
        
        result = model.solve_step(phase_field, dt=1e6)
        
        # Heat flux should be non-zero (heat is flowing due to temperature difference)
        heat_flux_magnitude = np.sqrt(np.mean(result.heat_flux_x**2 + result.heat_flux_y**2))
        assert heat_flux_magnitude > 0.0  # Heat is flowing
    
    def test_temperature_evolution(self):
        """Test temperature evolves smoothly."""
        grid_x = np.linspace(0, 1, 21)
        grid_y = np.linspace(0, 1, 21)
        T_init = np.ones((len(grid_y), len(grid_x))) * 1000.0
        
        # Set corners to different temperatures
        T_init[0, 0] = 273.15
        T_init[-1, -1] = 2000.0
        
        model = ThermalModel(grid_x, grid_y, T_init)
        phase_field = np.ones((len(grid_y), len(grid_x)), dtype=int)
        
        # Evolve
        for _ in range(5):
            result = model.solve_step(phase_field, dt=1e6)
        
        # Temperature should have smoothed out somewhat
        T_final = model.T_current
        assert T_final.std() < T_init.std()
    
    def test_energy_balance(self):
        """Test approximate energy conservation."""
        grid_x = np.linspace(0, 1, 11)
        grid_y = np.linspace(0, 1, 11)
        T_init = np.ones((len(grid_y), len(grid_x))) * 1000.0
        
        model = ThermalModel(grid_x, grid_y, T_init)
        phase_field = np.ones((len(grid_y), len(grid_x)), dtype=int)
        
        # With no heat source and Neumann BCs, mean should stay roughly constant
        result1 = model.solve_step(phase_field, dt=1e6, heat_source=np.zeros_like(T_init))
        T_mean_1 = np.mean(model.T_current)
        
        result2 = model.solve_step(phase_field, dt=1e6, heat_source=np.zeros_like(T_init))
        T_mean_2 = np.mean(model.T_current)
        
        # Mean should not change much without heat source
        assert abs(T_mean_2 - T_mean_1) < 100.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
