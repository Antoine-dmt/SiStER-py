---
agent: speckit.tasks
---

# Phase 0A: ConfigurationManager - Implementation Tasks

**Branch**: `001-configuration-manager`  
**Component**: ConfigurationManager  
**Estimated Duration**: 3-5 days  
**Status**: Ready for Implementation

---

## Overview

Implement the **ConfigurationManager** component for SiSteR-py: a production-grade YAML configuration loader with full validation, round-trip export, and material object creation.

**Why this matters**: ConfigurationManager is the **foundational component**—all other modules (Grid, Material, Solver, TimeStepper) depend on it. One YAML file drives the entire geodynamic simulation.

---

## Task 1: Project Setup & Dependencies

**Subtasks**:

1. Create `sister_py/` package directory structure
   ```
   sister_py/
   ├── __init__.py
   ├── config.py              ← Your main work
   ├── data/
   │   ├── examples/
   │   │   ├── continental_rift.yaml
   │   │   ├── subduction.yaml
   │   │   └── shear_flow.yaml
   │   └── defaults.yaml
   └── ...
   ```

2. Create `tests/test_config.py` (test-first: write tests before code)

3. Add dependencies to `pyproject.toml`:
   - `pyyaml>=6.0`
   - `pydantic>=2.0`
   - `python>=3.10`

4. Create example YAML files structure (can be filled in Task 4)

---

## Task 2: Pydantic Models & Validation (Test-First)

**Write tests first**, then implement:

### 2.1 Create all Pydantic BaseModel classes

```python
# sister_py/config.py

from pydantic import BaseModel, Field, field_validator, model_validator

class SimulationConfig(BaseModel):
    Nt: int = Field(..., gt=0)
    dt_out: int = Field(..., gt=0)
    output_dir: str

class DomainConfig(BaseModel):
    xsize: float = Field(..., gt=0)
    ysize: float = Field(..., gt=0)

class GridConfig(BaseModel):
    x_spacing: list[float]
    x_breaks: list[float]
    y_spacing: list[float]
    y_breaks: list[float]
    
    @field_validator('x_spacing', 'y_spacing')
    @classmethod
    def spacing_positive(cls, v):
        if any(x <= 0 for x in v):
            raise ValueError('Grid spacing must be positive')
        return v
    
    @field_validator('x_breaks', 'y_breaks')
    @classmethod
    def breaks_monotonic(cls, v):
        if not all(v[i] < v[i+1] for i in range(len(v)-1)):
            raise ValueError('Zone boundaries must be increasing')
        return v

class DensityParams(BaseModel):
    rho0: float = Field(..., gt=0)
    alpha: float

class DuctileCreepParams(BaseModel):
    A: float = Field(..., gt=0)
    E: float = Field(..., ge=0)
    n: float = Field(..., gt=0)

class RheologyConfig(BaseModel):
    type: str
    diffusion: DuctileCreepParams | None = None
    dislocation: DuctileCreepParams | None = None

class PlasticityParams(BaseModel):
    C: float = Field(..., ge=0)
    mu: float = Field(..., gt=0, lt=1)  # 0 < μ < 1

class ElasticityParams(BaseModel):
    G: float = Field(..., gt=0)

class ThermalParams(BaseModel):
    k: float
    cp: float

class MaterialConfig(BaseModel):
    phase: int = Field(..., gt=0)
    name: str
    density: DensityParams
    rheology: RheologyConfig | None = None
    plasticity: PlasticityParams | None = None
    elasticity: ElasticityParams | None = None
    thermal: ThermalParams | None = None

class BCConfig(BaseModel):
    type: str
    vx: float | None = None
    vy: float | None = None
    sxx: float | None = None
    sxy: float | None = None

class PhysicsConfig(BaseModel):
    elasticity: bool
    plasticity: bool
    thermal: bool

class SolverConfig(BaseModel):
    Npicard_min: int = Field(..., gt=0)
    Npicard_max: int = Field(..., gt=0)
    conv_tol: float = Field(..., gt=0)
    switch_to_newton: int = Field(..., ge=0)

class FullConfig(BaseModel):
    SIMULATION: SimulationConfig
    DOMAIN: DomainConfig
    GRID: GridConfig
    MATERIALS: list[MaterialConfig]
    BC: dict[str, BCConfig]
    PHYSICS: PhysicsConfig
    SOLVER: SolverConfig
    
    @field_validator('MATERIALS')
    @classmethod
    def phases_unique(cls, v):
        phases = [m.phase for m in v]
        if len(phases) != len(set(phases)):
            raise ValueError('Phase IDs must be unique')
        return v
```

### 2.2 Test Pydantic validation

Write tests in `tests/test_config.py`:
- Valid config loads without error
- Invalid `mu=1.5` rejected with granular message
- Invalid `x_spacing` negative rejected with zone identifier
- Multiple errors collected (not just first)
- All custom validators working

---

## Task 3: ConfigurationManager Class & Material Class

**Implement**:

```python
# sister_py/config.py

class Material:
    """Simple wrapper for material properties. No validation (done at config load)."""
    
    def __init__(self, config: MaterialConfig):
        self.config = config
    
    @property
    def phase(self) -> int:
        return self.config.phase
    
    @property
    def name(self) -> str:
        return self.config.name
    
    def density(self, T: float) -> float:
        """ρ(T) = ρ0 * (1 - α * ΔT), where ΔT = T - T_ref (assume T_ref ~ 0 K for now)"""
        rho0 = self.config.density.rho0
        alpha = self.config.density.alpha
        return rho0 * (1 - alpha * T)
    
    def viscosity_ductile(self, sigma_II: float, eps_II: float, T: float) -> float:
        """Power-law creep: ε̇ = A·σⁿ·exp(-E/RT)
        Invert to get η = σ / (2·ε̇)
        
        Args:
            sigma_II: Second invariant of deviatoric stress (Pa)
            eps_II: Second invariant of strain rate (1/s)
            T: Temperature (K)
        
        Returns:
            Viscosity (Pa·s)
        """
        if self.config.rheology is None:
            return float('inf')
        
        R = 8.314  # Gas constant (J/mol/K)
        
        # If both diffusion and dislocation, use harmonic mean
        etas = []
        
        if self.config.rheology.diffusion:
            p = self.config.rheology.diffusion
            A, E, n = p.A, p.E, p.n
            if sigma_II > 0 and eps_II > 0:
                # ε̇ = A·σⁿ·exp(-E/RT)
                # η = σ / (2·ε̇) = σ / (2·A·σⁿ·exp(-E/RT)) = 1/(2·A·σ^(n-1)·exp(-E/RT))
                eta_diff = 1 / (2 * A * (sigma_II ** (n - 1)) * np.exp(-E / (R * T)))
                etas.append(eta_diff)
        
        if self.config.rheology.dislocation:
            p = self.config.rheology.dislocation
            A, E, n = p.A, p.E, p.n
            if sigma_II > 0 and eps_II > 0:
                eta_disc = 1 / (2 * A * (sigma_II ** (n - 1)) * np.exp(-E / (R * T)))
                etas.append(eta_disc)
        
        if not etas:
            return float('inf')
        elif len(etas) == 1:
            return etas[0]
        else:
            # Harmonic mean
            return 1 / sum(1/eta for eta in etas)
    
    def viscosity_plastic(self, sigma_II: float, P: float) -> float:
        """Mohr-Coulomb yield: σ_Y = (C + μ·P)·cos(arctan(μ))
        Yield viscosity: η_yield = σ_Y / (2·ε̇)
        
        Args:
            sigma_II: Second invariant of stress (Pa)
            P: Pressure (Pa, positive for compression)
        
        Returns:
            Viscosity or inf if not yielding
        """
        if self.config.plasticity is None or P <= 0:
            return float('inf')
        
        C = self.config.plasticity.C
        mu = self.config.plasticity.mu
        
        sigma_Y = (C + mu * P) * np.cos(np.arctan(mu))
        
        if sigma_II > sigma_Y:
            # Yielding: cap viscosity at σ_Y / (2·ε̇)
            # But we need ε̇ ... use σ_Y directly for now
            return sigma_Y / 2  # Simplified
        else:
            return float('inf')
    
    def viscosity_effective(self, sigma_II: float, eps_II: float, T: float, P: float) -> float:
        """Effective viscosity: min(viscosity_ductile, viscosity_plastic)"""
        eta_duct = self.viscosity_ductile(sigma_II, eps_II, T)
        eta_plast = self.viscosity_plastic(sigma_II, P)
        return min(eta_duct, eta_plast)


class ConfigurationManager:
    """Load, validate, and provide access to simulation configuration."""
    
    def __init__(self, config: FullConfig):
        self.config = config
    
    @classmethod
    def load(cls, filepath: str) -> 'ConfigurationManager':
        """Load and validate YAML configuration file.
        
        Args:
            filepath: Path to YAML file (supports ${HOME} substitution)
        
        Returns:
            ConfigurationManager instance
        
        Raises:
            FileNotFoundError: If file doesn't exist
            ValidationError: If config invalid (Pydantic)
        """
        # Expand environment variables
        filepath = os.path.expandvars(filepath)
        
        # Load YAML
        with open(filepath, 'r') as f:
            data = yaml.safe_load(f)
        
        # Validate with Pydantic
        config = FullConfig(**data)
        
        return cls(config)
    
    def __getattr__(self, name):
        """Provide nested attribute access: cfg.DOMAIN.xsize"""
        if hasattr(self.config, name):
            return getattr(self.config, name)
        raise AttributeError(f"ConfigurationManager has no attribute '{name}'")
    
    def get_materials(self) -> dict[int, Material]:
        """Create Material objects from config MATERIALS section.
        
        Returns:
            dict mapping phase_id → Material instance
        """
        return {m.phase: Material(m) for m in self.config.MATERIALS}
    
    def to_dict(self) -> dict:
        """Return full config as nested dict (JSON-serializable)."""
        return self.config.model_dump()
    
    def to_yaml(self, filepath: str) -> None:
        """Save config to YAML file (round-trip safe, 6 sig figs precision)."""
        data = self.to_dict()
        with open(filepath, 'w') as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)
    
    def to_string(self) -> str:
        """Return formatted config as printable string (for stdout, logs)."""
        return yaml.dump(self.to_dict(), default_flow_style=False, sort_keys=False)
    
    def validate(self) -> None:
        """Re-validate current config (useful after programmatic modifications)."""
        FullConfig(**self.to_dict())
```

---

## Task 4: Create Example YAML Configurations

Create three example files:

### 4.1 `sister_py/data/examples/continental_rift.yaml`

```yaml
SIMULATION:
  Nt: 1600
  dt_out: 20
  output_dir: "./results"

DOMAIN:
  xsize: 170e3
  ysize: 60e3

GRID:
  x_spacing: [2000, 500, 2000]
  x_breaks: [50e3, 140e3]
  y_spacing: [2000, 500, 2000]
  y_breaks: [7e3, 40e3]

MATERIALS:
  - phase: 1
    name: "Sticky Layer"
    density:
      rho0: 1000
      alpha: 0
    rheology:
      type: "ductile"
      diffusion:
        A: 0.5e-18
        E: 0
        n: 1
      dislocation:
        A: 0.5e-18
        E: 0
        n: 1
    plasticity:
      C: 40e6
      mu: 0.6
    elasticity:
      G: 1e18
    thermal:
      k: 3
      cp: 1000

  - phase: 2
    name: "Mantle"
    density:
      rho0: 3300
      alpha: 3e-5
    rheology:
      type: "ductile"
      diffusion:
        A: 1e-21
        E: 400e3
        n: 3.5
      dislocation:
        A: 1.9e-16
        E: 540e3
        n: 3.5
    plasticity:
      C: 0
      mu: 0.6
    elasticity:
      G: 6.4e10
    thermal:
      k: 4.5
      cp: 1250

BC:
  top:
    type: "velocity"
    vx: 1e-10
    vy: 0
  bot:
    type: "velocity"
    vx: 0
    vy: 0
  left:
    type: "velocity"
    vx: 1e-10
    vy: 0
  right:
    type: "velocity"
    vx: 1e-10
    vy: 0

PHYSICS:
  elasticity: true
  plasticity: true
  thermal: true

SOLVER:
  Npicard_min: 10
  Npicard_max: 100
  conv_tol: 1e-9
  switch_to_newton: 0
```

### 4.2 `sister_py/data/examples/subduction.yaml`

Create subduction zone example (slab + mantle wedge)

### 4.3 `sister_py/data/examples/shear_flow.yaml`

Create simple shear flow test (easy to validate analytically)

### 4.4 `sister_py/data/defaults.yaml`

Default parameter values for all fields

---

## Task 5: Comprehensive Test Suite

**Write tests covering all acceptance criteria**:

### 5.1 Unit Tests (Pydantic Validation)
- Valid YAML loads ✅
- Invalid parameters rejected ✅
- Granular error messages ✅
- All errors collected ✅
- Custom validators working ✅

### 5.2 Round-Trip Tests
- Load → modify → save → load → identical ✅
- 6 sig figs maintained ✅
- Comments preserved ✅

### 5.3 Performance Tests
- Config load < 100 ms ✅
- Material access < 1 μs ✅

### 5.4 Integration Tests
- Material objects with viscosity methods ✅
- Nested attribute access ✅
- get_materials() returns correct dict ✅

### 5.5 Coverage
- Aim for > 90% coverage of config.py

---

## Task 6: Documentation & Examples

**Create docstrings & guide**:

1. **API Documentation** (in docstrings)
   - ConfigurationManager class
   - Material class
   - All Pydantic models

2. **"5-Minute Quick Start" guide**
   - Copy example YAML
   - Modify 3-4 parameters
   - Load and run

3. **YAML Schema Documentation**
   - Explain each section (SIMULATION, DOMAIN, GRID, MATERIALS, BC, PHYSICS, SOLVER)
   - Typical ranges for each parameter

---

## Acceptance Criteria (Binding)

✅ **All must pass**:

- [ ] Load valid YAML without error
- [ ] Reject invalid config with granular error message (not generic)
- [ ] Collect ALL validation errors (not just first)
- [ ] Round-trip: load → modify → save → load → bit-identical
- [ ] Performance: load 1000-line config < 100 ms
- [ ] Nested attribute access: `cfg.DOMAIN.xsize`
- [ ] Material objects: `cfg.get_materials()` returns dict
- [ ] Viscosity: matches MATLAB SiSteR to 6 sig figs
- [ ] Test coverage: > 90%
- [ ] All docstrings complete
- [ ] Examples provided & working

---

## Success Checklist

- [ ] Task 1: Project setup complete
- [ ] Task 2: Pydantic models with validation
- [ ] Task 3: ConfigurationManager & Material classes
- [ ] Task 4: Example YAML files (3 examples + defaults)
- [ ] Task 5: Comprehensive test suite (> 90% coverage)
- [ ] Task 6: Documentation & quick-start guide
- [ ] All tests pass ✅
- [ ] All acceptance criteria met ✅
- [ ] Constitution compliance verified ✅
- [ ] Ready for code review ✅

---

## Implementation Notes

- **Test-first**: Write tests before code (for each task)
- **Granular errors**: Use Pydantic validators for clear error messages
- **SI units**: All parameters in SI (K, Pa, Pa·s, J/mol, m, kg/m³)
- **Numba-ready**: Config data structures are NumPy-compatible (no Python objects)
- **Constitution compliance**: Follow 5 binding design principles

---

**Estimated Time**: 3-5 days  
**Branch**: `001-configuration-manager`  
**Ready**: Yes, proceed with Task 1 ✅
